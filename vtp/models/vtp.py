"""
Visual Tokenizer Pre-training.

A Framework capable of pre-training Vision Transformer with
Contrastive Image and Language Learning （CLIP or SigLIP style）
& Self-Supervised Learning (DINOv2 style)
& Reconstruction.
It demonstrates a brand-new scaling property for visual generation and understanding.

By Jingfeng Yao (Maple) from HUST-VL,
Work done during internship at MiniMax, 2025.

It is the legacay VTP Meta-Arch for CLIP+SSL+Reconstruction training.
In this repo, we use `vtp/models/vtp_hf/modeling_vtp.py` for fast inference.
"""

import copy
import logging
import os
from functools import partial
from types import SimpleNamespace
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .encoders import DinoVisionTransformerWithBottleneck, TextTransformer, text_global_pool
from .decoders import DinoV3PixelDecoder
from .heads import DINOHead
from .layers import LayerNormFp32, LayerNorm, QuickGELU


def get_cast_dtype(precision: str) -> Optional[torch.dtype]:
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str) -> Optional[torch.dtype]:
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_text_tower_from_config(vtp_config):
    quick_gelu = vtp_config.vtp_model.text_encoder.quick_gelu
    cast_dtype = vtp_config.training.cast_dtype
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    norm_kwargs = vtp_config.vtp_model.text_encoder.norm_kwargs
    act_kwargs = vtp_config.vtp_model.text_encoder.act_kwargs
    if norm_kwargs:
        norm_layer = partial(norm_layer, **norm_kwargs)
    if act_kwargs is not None:
        act_layer = partial(act_layer, **act_kwargs)

    text = TextTransformer(
        context_length=vtp_config.vtp_model.text_encoder.context_length,
        vocab_size=vtp_config.vtp_model.text_encoder.vocab_size,
        width=vtp_config.vtp_model.text_encoder.embed_dim,
        heads=vtp_config.vtp_model.text_encoder.heads,
        layers=vtp_config.vtp_model.text_encoder.layers,
        mlp_ratio=vtp_config.vtp_model.text_encoder.mlp_ratio,
        ls_init_value=vtp_config.vtp_model.text_encoder.ls_init_value,
        output_dim=vtp_config.vtp_model.text_encoder.embed_dim,
        embed_cls=vtp_config.vtp_model.text_encoder.embed_cls,
        no_causal_mask=vtp_config.vtp_model.text_encoder.no_causal_mask,
        pad_id=vtp_config.vtp_model.text_encoder.pad_id,
        pool_type=vtp_config.vtp_model.text_encoder.pool_type,
        proj_type=vtp_config.vtp_model.text_encoder.proj_type,
        proj_bias=vtp_config.vtp_model.text_encoder.proj_bias,
        output_tokens=vtp_config.vtp_model.text_encoder.output_tokens,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return text


class VTP(nn.Module):
    """Visual Tokenizer Pre-training Model.

    A unified framework for vision-language pre-training that supports:
    - Contrastive learning (CLIP-style)
    - Self-supervised learning (DINOv2-style)
    - Image reconstruction
    """

    def __init__(
        self,
        vtp_config: Optional[DictConfig] = None,
        config_path: Optional[str] = None,
        cli_overrides: Optional[Union[Sequence[str], DictConfig]] = None,
    ):
        """
        Args:
            vtp_config: Pre-constructed configuration (takes priority).
            config_path: Path to load YAML from when vtp_config is not provided.
            cli_overrides: Additional CLI-style configuration overrides, e.g., ["training.lr=1e-4"].
        """
        super().__init__()

        self.vtp_config = self._load_vtp_config(
            vtp_config=vtp_config,
            config_path=config_path,
            cli_overrides=cli_overrides,
        )
        self._init_vision_components()
        self._init_text_components()

    def _load_vtp_config(
        self,
        vtp_config: Optional[DictConfig],
        config_path: Optional[str],
        cli_overrides: Optional[Union[Sequence[str], DictConfig]],
    ) -> DictConfig:
        if config_path is not None and not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)

        if vtp_config is not None:
            if isinstance(vtp_config, DictConfig):
                loaded_config = vtp_config
            else:
                cfg_dict = self._object_to_config_dict(vtp_config)
                loaded_config = OmegaConf.create(cfg_dict)
        else:
            if config_path is None:
                raise ValueError("Either vtp_config or config_path must be provided")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            loaded_config = OmegaConf.load(config_path)

        if cli_overrides:
            if isinstance(cli_overrides, DictConfig):
                overrides_cfg = cli_overrides
            else:
                if isinstance(cli_overrides, (list, tuple)):
                    overrides_list = list(cli_overrides)
                else:
                    overrides_list = [cli_overrides]
                overrides_cfg = OmegaConf.from_cli(overrides_list)
            loaded_config = OmegaConf.merge(loaded_config, overrides_cfg)

        return loaded_config

    def _object_to_config_dict(self, obj):
        if isinstance(obj, DictConfig):
            return obj
        if isinstance(obj, SimpleNamespace):
            obj = vars(obj)
        if isinstance(obj, dict):
            return {k: self._object_to_config_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._object_to_config_dict(v) for v in obj]
        return obj

    def _init_text_components(self):

        if self.vtp_config.training.train_clip:
            text = _build_text_tower_from_config(self.vtp_config)
            self.transformer = text.transformer
            self.context_length = text.context_length
            self.vocab_size = text.vocab_size
            self.token_embedding = text.token_embedding
            self.positional_embedding = text.positional_embedding
            self.ln_final = text.ln_final
            self.text_projection = text.text_projection
            self.text_pool_type = text.pool_type
            self.register_buffer('attn_mask', text.attn_mask, persistent=False)

            init_logit_scale = self.vtp_config.training.init_logit_scale or np.log(1 / 0.07)
            init_logit_bias = self.vtp_config.training.init_logit_bias
            nonscalar_logit_scale = self.vtp_config.training.nonscalar_logit_scale

            lshape = [1] if nonscalar_logit_scale else []
            self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
            if init_logit_bias is not None:
                self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
            else:
                self.logit_bias = None
            self.output_dict = self.vtp_config.training.clip_output_dict
        else:
            logging.info("Not training clip")


    def _init_vision_components(self):

        vit_kwargs = self.vtp_config.vtp_model.vision_encoder
        vit_kwargs['img_size'] = self.vtp_config.data.image_size
        vit_kwargs['use_mask_token'] = self.vtp_config.training.train_dinov2

        if self.vtp_config.vtp_model.vision_encoder.model_type == 'dinov3':
            vit_kwargs['norm_layer'] = vit_kwargs.pop('norm_type')
            vit_kwargs['ffn_ratio'] = vit_kwargs.pop('mlp_ratio')
            vit_kwargs['layerscale_init'] = vit_kwargs.pop('init_values')
            self.trunk = DinoVisionTransformerWithBottleneck(**vit_kwargs)
            # Allow dropout for DINOv3 runtime forward (clip/ssl/rec)
            self.clip_drop_rate = self.vtp_config.training.clip_drop_rate
            self.ssl_drop_rate = self.vtp_config.training.ssl_drop_rate
            self.rec_drop_rate = self.vtp_config.training.rec_drop_rate
            logging.info(
                f"DINOv3 drop configured - clip: {self.clip_drop_rate}, ssl(student): {self.ssl_drop_rate}, rec: {self.rec_drop_rate}; ssl(teacher): 0.0"
            )
        else:
            raise ValueError(f"Unsupported vision encoder: {self.vtp_config.vtp_model.vision_encoder.model_type}")

        effective_embed_dim = self.trunk.vit_feature_bottleneck
        if self.vtp_config.training.train_clip:
            self.proj = nn.Linear(effective_embed_dim if not self.vtp_config.vtp_model.vision_encoder.bottleneck_ae_only else self.trunk.embed_dim, self.vtp_config.vtp_model.text_encoder.embed_dim, bias=False)
        else:
            self.proj = None

        if self.vtp_config.training.train_dinov2:
            self.dino_head = DINOHead(
                in_dim=effective_embed_dim if not self.vtp_config.vtp_model.vision_encoder.bottleneck_ae_only else self.trunk.embed_dim,
                out_dim=self.vtp_config.vtp_model.dino_head.out_dim,
                nlayers=self.vtp_config.vtp_model.dino_head.nlayers,
                hidden_dim=self.vtp_config.vtp_model.dino_head.hidden_dim,
                bottleneck_dim=self.vtp_config.vtp_model.dino_head.bottleneck_dim,
            )
        else:
            self.dino_head = None

        if self.vtp_config.training.train_reconstruction:
            decoder_kwargs = self.vtp_config.vtp_model.pixel_decoder
            if self.vtp_config.vtp_model.pixel_decoder.model_type == 'dinov3':
                decoder_kwargs["in_chans"] = effective_embed_dim
                self.pixel_decoder = DinoV3PixelDecoder(
                    **decoder_kwargs
                )
            else:
                raise ValueError(f"Unsupported pixel decoder: {self.vtp_config.vtp_model.pixel_decoder.model_type}")
        else:
            self.pixel_decoder = None

        if self.vtp_config.training.train_dinov2:
            self.enable_teacher = True
            with torch.no_grad():
                self.teacher_trunk = copy.deepcopy(self.trunk)
                if self.vtp_config.training.train_clip:
                    self.teacher_proj = nn.Linear(
                        effective_embed_dim if not self.vtp_config.vtp_model.vision_encoder.bottleneck_ae_only else self.trunk.embed_dim,
                        self.vtp_config.vtp_model.text_encoder.embed_dim, bias=False
                        )
                else:
                    self.teacher_proj = None
                self.teacher_dino_head = DINOHead(
                    in_dim=effective_embed_dim if not self.vtp_config.vtp_model.vision_encoder.bottleneck_ae_only else self.trunk.embed_dim,
                    out_dim=self.vtp_config.vtp_model.dino_head.out_dim,
                    nlayers=self.vtp_config.vtp_model.dino_head.nlayers,
                    hidden_dim=self.vtp_config.vtp_model.dino_head.hidden_dim,
                    bottleneck_dim=self.vtp_config.vtp_model.dino_head.bottleneck_dim,
                )
                for p in self.teacher_trunk.parameters():
                    p.requires_grad = False
                if self.vtp_config.training.train_clip:
                    for p in self.teacher_proj.parameters():
                        p.requires_grad = False
                for p in self.teacher_dino_head.parameters():
                    p.requires_grad = False
        else:
            self.teacher_trunk = None
            self.teacher_proj = None
            self.teacher_dino_head = None
            self.enable_teacher = False

    def encode_image(self, image: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        x_dict = self.trunk(
            image,
            is_training=True,
            use_bottleneck=not self.vtp_config.vtp_model.vision_encoder.bottleneck_ae_only,
            drop_ratio=self.clip_drop_rate,
        )
        pooled, tokens = x_dict['x_norm_clstoken'], x_dict['x_norm_patchtokens']

        if self.vtp_config.vtp_model.vision_encoder.clip_feat == 'cls':
            features = pooled
        elif self.vtp_config.vtp_model.vision_encoder.clip_feat == 'pooled':
            features = tokens.mean(dim=1)
        else:
            raise ValueError(f"Invalid clip feat: {self.vtp_config.vtp_model.vision_encoder.clip_feat}")

        features = self.proj(features)

        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)

        x = text_global_pool(x, text, self.text_pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_logits(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        ssl_dict: Optional[dict] = None,
        reconstruction_image: Optional[torch.Tensor] = None,
        forward_type: str = "clip",
    ):
        assert forward_type in ["clip", "ssl", "rec"], "Invalid forward type"

        if forward_type == "clip":
            return self.forward_clip(image, text)
        elif forward_type == "ssl":
            return self.forward_ssl_learning(**ssl_dict)
        elif forward_type == "rec":
            return self.forward_reconstruction(reconstruction_image)

    def forward_clip(
        self,
        image: Optional[torch.Tensor],
        text: Optional[torch.Tensor]
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()

    def forward_reconstruction(self, reconstruction_image: torch.Tensor):
        return self.get_reconstruction_outputs(reconstruction_image)

    def forward_ssl_learning(
        self,
        global_crops,
        n_global_crops,
        mask_indices_list,
        n_masked_patches,
        upperbound,
        local_crops,
        masks
    ):
        teacher_outputs = self.get_teacher_forward_outputs(
            global_crops, n_global_crops, mask_indices_list,
            n_masked_patches, upperbound
        )

        student_outputs = self.get_student_ssl_outputs(
            global_crops, local_crops, masks, mask_indices_list,
            n_masked_patches, upperbound
        )

        return teacher_outputs, student_outputs


    def update_teacher(self, momentum: float):
        if not self.enable_teacher:
            return

        with torch.no_grad():
            for student_param, teacher_param in zip(self.trunk.parameters(), self.teacher_trunk.parameters()):
                teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data

            if self.vtp_config.training.train_clip:
                for student_param, teacher_param in zip(self.proj.parameters(), self.teacher_proj.parameters()):
                    teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data

            for student_param, teacher_param in zip(self.dino_head.parameters(), self.teacher_dino_head.parameters()):
                teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data

    def get_ssl_params(self):
        ssl_params = []
        if self.enable_teacher and self.dino_head is not None:
            ssl_params.extend(list(self.dino_head.parameters()))
        return ssl_params


    def get_teacher_forward_outputs(self, global_crops, n_global_crops, mask_indices_list,
                                  n_masked_patches, upperbound):
        if not self.enable_teacher:
            return {}

        teacher_outputs = {}

        with torch.no_grad():
            teacher_backbone_output_dict = self.teacher_trunk(
                global_crops,
                is_training=True,
                use_bottleneck=not self.vtp_config.vtp_model.vision_encoder.bottleneck_ae_only,
                drop_ratio=0.0
            )
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops)
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))

            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
            buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
            torch.index_select(
                ibot_teacher_patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
            )
            tokens_after_head = self.teacher_dino_head(buffer_tensor_teacher)
            teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
            masked_teacher_patch_tokens_after_head = tokens_after_head[
                n_cls_tokens : n_cls_tokens + n_masked_patches
            ]

            teacher_outputs['teacher_cls_tokens_after_head'] = teacher_cls_tokens_after_head
            teacher_outputs['n_masked_patches'] = n_masked_patches
            teacher_outputs['masked_teacher_patch_tokens_after_head'] = masked_teacher_patch_tokens_after_head

        return teacher_outputs

    def get_student_ssl_outputs(self, global_crops, local_crops, masks, mask_indices_list,
                               n_masked_patches, upperbound):
        if not self.enable_teacher:
            return {}

        student_global_output, student_local_output = self.trunk(
            [global_crops, local_crops],
            is_training=True,
            masks=[masks, None],
            use_bottleneck=not self.vtp_config.vtp_model.vision_encoder.bottleneck_ae_only,
            drop_ratio=self.ssl_drop_rate,
        )

        student_local_cls_tokens = student_local_output["x_norm_clstoken"]
        student_global_cls_tokens = student_global_output["x_norm_clstoken"]

        ibot_student_patch_tokens = student_global_output["x_norm_patchtokens"]
        _dim = ibot_student_patch_tokens.shape[-1]
        buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
        buffer_tensor_patch_tokens[:n_masked_patches].copy_(
            torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
        )

        student_local_cls_tokens_after_head = self.dino_head(student_local_cls_tokens)
        student_global_cls_tokens_after_head = self.dino_head(student_global_cls_tokens)
        student_global_masked_patch_tokens_after_head = self.dino_head(buffer_tensor_patch_tokens)[:n_masked_patches]

        return {
            'student_local_cls_tokens_after_head': student_local_cls_tokens_after_head,
            'student_global_cls_tokens_after_head': student_global_cls_tokens_after_head,
            'student_global_cls_tokens': student_global_cls_tokens,
            'student_global_masked_patch_tokens_after_head': student_global_masked_patch_tokens_after_head,
        }


    def get_reconstruction_outputs(self, reconstruction_image):
        if not self.vtp_config.training.train_reconstruction:
            return {}

        _, _, H, W = reconstruction_image.shape
        self._current_img_h = H
        self._current_img_w = W

        student_output = self.trunk(
            reconstruction_image,
            is_training=True,
            masks=None,
            use_bottleneck=True,
            drop_ratio=self.rec_drop_rate,
        )

        # Convert 3D patch tokens (B, N, C) to 4D format (B, C, H, W)
        patch_tokens = student_output['x_norm_patchtokens']  # (B, N, C)
        patch_tokens_4d = self._convert_patch_tokens_to_4d(patch_tokens)

        pixel_rec = self.pixel_decoder(patch_tokens_4d)

        return {
            'reconstructed_image': pixel_rec,
            'target_image': reconstruction_image
        }

    def _convert_patch_tokens_to_4d(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert 3D patch tokens (B, N, C) to 4D format (B, C, H, W) for pixel decoder.

        Args:
            patch_tokens: Patch tokens from DINOv2 with shape (B, N, C)

        Returns:
            torch.Tensor: Reshaped tokens with shape (B, C, H, W)
        """
        B, N, C = patch_tokens.shape

        # Use recorded original image dimensions instead of preset values
        patch_size = self.vtp_config.vtp_model.vision_encoder.patch_size

        # Get actual image dimensions from recorded values
        if hasattr(self, '_current_img_h') and hasattr(self, '_current_img_w'):
            img_h, img_w = self._current_img_h, self._current_img_w
        else:
            # Fallback to config values if not recorded
            logging.warning("No recorded image dimensions found, falling back to config values")
            img_h = img_w = self.vtp_config.data.image_size

        # Calculate feature map dimensions based on actual image size
        feat_h = img_h // patch_size
        feat_w = img_w // patch_size

        # Verify that N matches expected number of patches
        expected_patches = feat_h * feat_w
        if N != expected_patches:
            raise ValueError(
                f"Patch tokens dimension mismatch: got {N} patches, "
                f"expected {expected_patches} (actual_img={img_h}x{img_w}, patch_size={patch_size}, feat_grid={feat_h}x{feat_w})"
            )

        # Reshape from (B, N, C) to (B, C, H, W)
        patch_tokens_4d = patch_tokens.transpose(1, 2).reshape(B, C, feat_h, feat_w)

        return patch_tokens_4d
