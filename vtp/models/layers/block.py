from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ..utils import cat_keep_shapes, uncat_with_shapes
from .attention import Attention, CausalSelfAttention, SelfAttention
from .ffn import Mlp
from .misc import LayerScale
from .normalization import LayerNorm

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


_GLOBAL_SAMPLING_CACHE: dict = {}


def get_branges_scales(x: Tensor, sample_drop_ratio: float = 0.0):
    """
    Global sampling allocation aligned with dinov2:
    - Under DDP, compute global keep samples and distribute evenly across ranks
    - Return local indices (brange) and global residual_scale_factor
    """
    b = x.shape[0]

    try:
        import torch.distributed as dist
        ddp_inited = dist.is_available() and dist.is_initialized()
    except Exception:
        ddp_inited = False

    if not ddp_inited:
        exact_keep = b * (1 - sample_drop_ratio)
        keep = max(int(exact_keep), 1)
        brange = torch.randperm(b, device=x.device)[:keep]
        scale = b / keep
        return brange, scale

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    global_batch = b * world_size
    cache_key = (global_batch, float(sample_drop_ratio), world_size)

    if cache_key not in _GLOBAL_SAMPLING_CACHE:
        if rank == 0:
            print(f"[Sampling Cache] New configuration detected:")
            print(f"  - Global batch size: {global_batch}")
            print(f"  - Sample drop ratio: {sample_drop_ratio:.3f}")
            print(f"  - World size: {world_size}")
            print(f"  - Per-GPU batch size: {b}")

            global_exact_keep = global_batch * (1 - sample_drop_ratio)
            global_keep = max(int(global_exact_keep), world_size)

            print(f"  - Global keep samples: {global_keep} (exact: {global_exact_keep:.2f})")

            base = global_keep // world_size
            extra = global_keep % world_size
            allocation = []
            for i in range(world_size):
                n_keep = base + (1 if i < extra else 0)
                n_keep = min(n_keep, b)
                allocation.append(n_keep)

            actual_global_keep = sum(allocation)
            residual_scale_factor = global_batch / max(actual_global_keep, 1)

            allocation_info = {
                'allocation': allocation,
                'global_keep_samples': global_keep,
                'actual_global_samples': actual_global_keep,
                'residual_scale_factor': residual_scale_factor,
            }

            print(f"  - Allocation per GPU: {allocation}")
            print(f"  - Actual global samples: {actual_global_keep}")
            print(f"  - Residual scale factor: {residual_scale_factor:.4f}")
            print(f"  - Saving to cache...")

            _GLOBAL_SAMPLING_CACHE[cache_key] = allocation_info

            payload = torch.tensor([
                allocation_info['global_keep_samples'],
                allocation_info['actual_global_samples'],
                allocation_info['residual_scale_factor']
            ] + allocation_info['allocation'], dtype=torch.float32, device=x.device)

            print(f"  - Broadcasting allocation to all nodes...")
        else:
            payload = torch.zeros(3 + world_size, dtype=torch.float32, device=x.device)
        dist.broadcast(payload, src=0)
        if rank != 0:
            global_keep = int(payload[0].item())
            actual_global_keep = int(payload[1].item())
            residual_scale_factor = float(payload[2].item())
            allocation = [int(payload[3 + i].item()) for i in range(world_size)]

            allocation_info = {
                'allocation': allocation,
                'global_keep_samples': global_keep,
                'actual_global_samples': actual_global_keep,
                'residual_scale_factor': residual_scale_factor,
            }

            _GLOBAL_SAMPLING_CACHE[cache_key] = allocation_info
            print(f"[Sampling Cache] Rank {rank} received allocation: {allocation}")

    info = _GLOBAL_SAMPLING_CACHE[cache_key]
    local_keep = int(info['allocation'][rank])
    residual_scale_factor = float(info['residual_scale_factor'])
    if local_keep > 0:
        brange = torch.randperm(b, device=x.device)[:local_keep]
    else:
        brange = torch.empty(0, dtype=torch.long, device=x.device)
    return brange, residual_scale_factor


def clear_sampling_cache():
    global _GLOBAL_SAMPLING_CACHE
    _GLOBAL_SAMPLING_CACHE.clear()
    print("[Sampling Cache] Cache cleared")


def print_sampling_cache_info():
    global _GLOBAL_SAMPLING_CACHE
    print(f"[Sampling Cache] Current cache size: {len(_GLOBAL_SAMPLING_CACHE)}")
    for i, (key, info) in enumerate(_GLOBAL_SAMPLING_CACHE.items()):
        global_batch, drop_ratio, world_size = key
        print(f"  Cache [{i+1}]: global_batch={global_batch}, drop_ratio={drop_ratio:.3f}, world_size={world_size}")
        print(f"    Allocation: {info['allocation']}")
        print(f"    Scale factor: {info['residual_scale_factor']:.4f}")


class SelfAttentionBlock(nn.Module):
    """Self-attention block with FFN for DINOv3."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
            use_qk_norm=use_qk_norm,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: Optional[Tuple[Tensor, Tensor]], indices: Tensor) -> Optional[Tuple[Tensor, Tensor]]:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            return sin[indices], cos[indices]
        else:
            return sin, cos

    def _forward(self, x: Tensor, rope=None, drop_ratio: Optional[float] = None) -> Tensor:
        b, _, _ = x.shape
        effective_drop_ratio = drop_ratio if drop_ratio is not None else self.sample_drop_ratio
        if self.training and effective_drop_ratio > 0.0:
            indices_1, residual_scale_factor = get_branges_scales(x, effective_drop_ratio)

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )
            indices_2, residual_scale_factor = get_branges_scales(x_attn, effective_drop_ratio)
            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None, drop_ratio: Optional[float] = None) -> List[Tensor]:
        b_list = [x.shape[0] for x in x_list]
        effective_drop_ratio = drop_ratio if drop_ratio is not None else self.sample_drop_ratio
        if self.training and effective_drop_ratio > 0.0:
            branges_scales_1 = [get_branges_scales(x, effective_drop_ratio) for x in x_list]
            indices_1_list = [br for br, _ in branges_scales_1]
            residual_scale_factors = [sc for _, sc in branges_scales_1]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            branges_scales_2 = [get_branges_scales(x_attn, effective_drop_ratio) for x_attn in x_attn_list]
            indices_2_list = [br for br, _ in branges_scales_2]
            residual_scale_factors_2 = [sc for _, sc in branges_scales_2]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors_2
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None, drop_ratio: Optional[float] = None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list], drop_ratio=drop_ratio)[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list, drop_ratio=drop_ratio)
        else:
            raise AssertionError


class CausalSelfAttentionBlock(nn.Module):
    """Causal self-attention block for autoregressive models."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: Optional[float] = None,
        init_proj_std: Optional[float] = None,
        init_fc_std: Optional[float] = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ):

        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn


class ResidualAttentionBlock(nn.Module):
    """Residual attention block for CLIP-style transformers."""

    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    """Custom residual attention block with scaled cosine attention."""

    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            batch_first=batch_first,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_reference_weight(self):
        return self.mlp.c_fc.weight

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x
