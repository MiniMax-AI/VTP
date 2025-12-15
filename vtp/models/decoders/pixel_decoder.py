import logging
from typing import Any, Literal, Optional

import torch
import torch.nn.init
from torch import Tensor, nn

from ..layers import RopePositionEmbedding, SelfAttentionBlock
from ..encoders import ffn_layer_dict, norm_layer_dict, dtype_dict, init_weights_vit
from ..utils import named_apply

logger = logging.getLogger("dinov3_pixel_decoder")


class DinoV3PixelDecoder(nn.Module):
    """DINOv3-based Pixel Decoder for image reconstruction.

    This decoder uses a transformer architecture with RoPE position embeddings
    and PixelShuffle for upsampling latent features back to pixel space.
    """

    def __init__(
        self,
        *,
        in_chans: int = 256,
        out_chans: int = 3,
        upscale_factor: int = 16,
        # ViT params
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: Optional[float] = None,
        pos_embed_rope_max_period: Optional[float] = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: Optional[float] = None,
        pos_embed_rope_jitter_coords: Optional[float] = None,
        pos_embed_rope_rescale_coords: Optional[float] = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: Optional[float] = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "swiglu",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        mask_k_bias: bool = False,
        device: Optional[Any] = None,
        use_qk_norm: bool = False,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. Input projection
        self.proj_in = nn.Conv2d(
            in_chans, embed_dim, kernel_size=1, bias=proj_bias
        )

        # 2. RoPE
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )

        # 3. Transformer blocks
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio_sequence[i],
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer_cls,
                    act_layer=nn.GELU,
                    ffn_layer=ffn_layer_cls,
                    init_values=layerscale_init,
                    mask_k_bias=mask_k_bias,
                    device=device,
                    use_qk_norm=use_qk_norm,
                )
                for i in range(depth)
            ]
        )

        # 4. Final norm
        self.norm = norm_layer_cls(embed_dim)

        # 5. Output projection for Pixel Shuffle
        self.upscale_factor = upscale_factor
        proj_out_chans = out_chans * (self.upscale_factor**2)
        self.proj_out = nn.Conv2d(
            embed_dim, proj_out_chans, kernel_size=1, bias=proj_bias
        )

        # 6. Pixel Shuffle layer
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)

        # Weight init
        self.init_weights()

    def init_weights(self):
        # Initialize proj_in and proj_out with trunc_normal_
        for m in [self.proj_in, self.proj_out]:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize transformer blocks
        named_apply(init_weights_vit, self)
        # RoPE does not need init here, it's handled internally

    def forward(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape

        # 1. Project in: (B, C, H, W) -> (B, D, H, W)
        x = self.proj_in(x)

        # 2. Reshape for transformer: (B, D, H, W) -> (B, H*W, D)
        x = x.flatten(2).transpose(1, 2)

        # 3. Get RoPE:
        rope_sincos = self.rope_embed(H=H, W=W)

        # 4. Transformer blocks
        for blk in self.blocks:
            x = blk(x, rope_sincos)

        # 5. Final Norm
        x = self.norm(x)

        # 6. Reshape back to image-like: (B, H*W, D) -> (B, D, H, W)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        # 7. Project out: (B, D, H, W) -> (B, C_out * up_factor^2, H, W)
        x = self.proj_out(x)

        # 8. Pixel Shuffle: (B, C_out * up_factor^2, H, W) -> (B, C_out, H*up_factor, W*up_factor)
        x = self.pixel_shuffle(x)

        return x


# Factory functions for different model sizes
def dinov3_pixel_decoder_small(
    in_chans=256, out_chans=3, upscale_factor=4, **kwargs
):
    """Small DINOv3-based pixel decoder"""
    model = DinoV3PixelDecoder(
        in_chans=in_chans,
        out_chans=out_chans,
        upscale_factor=upscale_factor,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def dinov3_pixel_decoder_base(
    in_chans=256, out_chans=3, upscale_factor=4, **kwargs
):
    """Base DINOv3-based pixel decoder"""
    model = DinoV3PixelDecoder(
        in_chans=in_chans,
        out_chans=out_chans,
        upscale_factor=upscale_factor,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def dinov3_pixel_decoder_large(
    in_chans=256, out_chans=3, upscale_factor=4, **kwargs
):
    """Large DINOv3-based pixel decoder"""
    model = DinoV3PixelDecoder(
        in_chans=in_chans,
        out_chans=out_chans,
        upscale_factor=upscale_factor,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model
