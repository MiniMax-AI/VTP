"""
Visual Tokenizer Pre-training.

A Framework capable of pre-training Vision Transformer with
Contrastive Image and Language Learning （CLIP or SigLIP style）
& Self-Supervised Learning (DINOv2 style)
& Reconstruction.
It demonstrates a brand-new scaling property for visual generation and understanding.

By Jingfeng Yao (Maple) from HUST-VL,
Work done during internship at MiniMax, 2025.
"""

import logging
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from .configuration_vtp import VTPConfig
from ..encoders import DinoVisionTransformerWithBottleneck, TextTransformer, text_global_pool
from ..decoders import DinoV3PixelDecoder
from ..layers import LayerNorm, QuickGELU

logger = logging.getLogger(__name__)


class VTPPreTrainedModel(PreTrainedModel):
    """Base class for VTP models."""

    config_class = VTPConfig
    base_model_prefix = "vtp"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize weights following VTP conventions."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


class VTPModel(VTPPreTrainedModel):
    """VTP (Visual Tokenizer Pre-training) Model.

    A unified framework supporting multiple vision tasks through composable methods:

    Basic feature extraction:
        - get_last_layer_feature(): Raw features from last layer
        - get_intermediate_layers_feature(): Multi-layer features

    CLIP zero-shot:
        - get_clip_image_feature(): CLIP-projected image features
        - get_clip_text_feature(): CLIP-projected text features

    Reconstruction:
        - get_reconstruction_latents(): Bottleneck latents for reconstruction
        - get_latents_decoded_images(): Decode latents to images

    Example:
        >>> config = VTPConfig(vision_embed_dim=768, train_clip=True)
        >>> model = VTPModel(config)
        >>>
        >>> # CLIP zero-shot
        >>> img_feat = model.get_clip_image_feature(images)
        >>> txt_feat = model.get_clip_text_feature(tokens)
        >>>
        >>> # Reconstruction
        >>> latents = model.get_reconstruction_latents(images)
        >>> reconstructed = model.get_latents_decoded_images(latents)
        >>>
    """

    def __init__(self, config: VTPConfig):
        super().__init__(config)
        self.config = config

        self._init_vision_components()
        if config.train_clip:
            self._init_text_components()

        self.post_init()

    def _init_vision_components(self):
        """Initialize vision encoder and related components."""
        config = self.config

        # Vision encoder
        self.trunk = DinoVisionTransformerWithBottleneck(
            img_size=config.image_size,
            patch_size=config.vision_patch_size,
            embed_dim=config.vision_embed_dim,
            depth=config.vision_depth,
            num_heads=config.vision_num_heads,
            ffn_ratio=config.vision_mlp_ratio,
            ffn_layer=config.vision_ffn_layer,
            norm_layer=config.vision_norm_layer,
            layerscale_init=config.vision_init_values,
            use_qk_norm=config.vision_use_qk_norm,
            vit_feature_bottleneck=config.vision_feature_bottleneck,
        )

        effective_embed_dim = self.trunk.vit_feature_bottleneck

        # CLIP projection
        if config.train_clip:
            proj_in_dim = effective_embed_dim if not config.vision_bottleneck_ae_only else self.trunk.embed_dim
            self.visual_proj = nn.Linear(proj_in_dim, config.text_embed_dim, bias=False)
        else:
            self.visual_proj = None

        # Pixel decoder
        if config.train_reconstruction:
            self.pixel_decoder = DinoV3PixelDecoder(
                in_chans=effective_embed_dim,
                embed_dim=config.decoder_embed_dim,
                num_heads=config.decoder_num_heads,
                depth=config.decoder_depth,
                ffn_layer=config.decoder_ffn_layer,
                norm_layer=config.decoder_norm_layer,
                layerscale_init=config.decoder_init_values,
                use_qk_norm=config.decoder_use_qk_norm,
            )
        else:
            self.pixel_decoder = None

    def _init_text_components(self):
        """Initialize text encoder and related components."""
        config = self.config

        act_layer = QuickGELU if config.text_quick_gelu else nn.GELU
        norm_layer = LayerNorm

        text = TextTransformer(
            context_length=config.text_context_length,
            vocab_size=config.text_vocab_size,
            width=config.text_embed_dim,
            heads=config.text_num_heads,
            layers=config.text_depth,
            mlp_ratio=config.text_mlp_ratio,
            ls_init_value=config.text_ls_init_value,
            output_dim=config.text_embed_dim,
            embed_cls=config.text_embed_cls,
            no_causal_mask=config.text_no_causal_mask,
            pad_id=config.text_pad_id,
            pool_type=config.text_pool_type,
            proj_type=config.text_proj_type,
            proj_bias=config.text_proj_bias,
            output_tokens=config.text_output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.text_transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        # Logit scale
        init_logit_scale = config.init_logit_scale or np.log(1 / 0.07)
        lshape = [1] if config.nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)

        if config.init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * config.init_logit_bias)
        else:
            self.logit_bias = None

    # ==================== Basic Feature Methods ====================

    def get_last_layer_feature(
        self,
        image: torch.Tensor,
        use_bottleneck: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Get features from the last layer of vision encoder.

        This returns the original high-dimensional features without bottleneck,
        which is typically desired for SSL tasks like linear probing.

        Args:
            image: Input images, shape (B, C, H, W).
            use_bottleneck: Whether to apply feature bottleneck. Default False
                for SSL feature extraction; set True for reconstruction.

        Returns:
            Dict with keys:
                - 'cls_token': CLS token features, shape (B, D)
                - 'patch_tokens': Patch token features, shape (B, N, D)
        """
        output = self.trunk(
            image,
            is_training=True,  # Must be True to get full dict output
            use_bottleneck=use_bottleneck,
        )
        return {
            'cls_token': output['x_norm_clstoken'],
            'patch_tokens': output['x_norm_patchtokens'],
        }

    def get_intermediate_layers_feature(
        self,
        image: torch.Tensor,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """Get features from intermediate layers.

        Args:
            image: Input images, shape (B, C, H, W).
            n: Number of last layers to return, or list of layer indices.
            reshape: If True, reshape to (B, C, H, W) format.
            return_class_token: If True, also return class tokens.
            norm: If True, apply layer normalization.

        Returns:
            Tuple of intermediate features.
        """
        return self.trunk.get_intermediate_layers(
            image,
            n=n,
            reshape=reshape,
            return_class_token=return_class_token,
            norm=norm,
        )

    # ==================== CLIP Methods ====================

    def get_clip_image_feature(
        self,
        image: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Get CLIP-projected image features for zero-shot tasks.

        Args:
            image: Input images, shape (B, C, H, W).
            normalize: Whether to L2 normalize output.

        Returns:
            Image features, shape (B, D).
        """
        if self.visual_proj is None:
            raise RuntimeError("CLIP not enabled. Set train_clip=True in config.")

        output = self.trunk(
            image,
            is_training=True,  # Must be True to get full dict output
            use_bottleneck=not self.config.vision_bottleneck_ae_only,
        )

        if self.config.vision_clip_feat == 'cls':
            features = output['x_norm_clstoken']
        elif self.config.vision_clip_feat == 'pooled':
            features = output['x_norm_patchtokens'].mean(dim=1)
        else:
            raise ValueError(f"Invalid vision_clip_feat: {self.config.vision_clip_feat}")

        features = self.visual_proj(features)

        return F.normalize(features, dim=-1) if normalize else features

    def get_clip_text_feature(
        self,
        text: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Get CLIP-projected text features for zero-shot tasks.

        Args:
            text: Input token IDs, shape (B, L).
            normalize: Whether to L2 normalize output.

        Returns:
            Text features, shape (B, D).
        """
        if not self.config.train_clip:
            raise RuntimeError("CLIP not enabled. Set train_clip=True in config.")

        cast_dtype = self.text_transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = self.text_transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)

        x = text_global_pool(x, text, self.text_pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_clip_logits(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CLIP similarity logits.

        Args:
            image: Input images, shape (B_img, C, H, W).
            text: Input token IDs, shape (B_txt, L).

        Returns:
            Tuple of (image_to_text_logits, text_to_image_logits).
        """
        image_features = self.get_clip_image_feature(image, normalize=True)
        text_features = self.get_clip_text_feature(text, normalize=True)

        logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            logits = logits + self.logit_bias

        return logits, logits.T

    # ==================== Reconstruction Methods ====================

    def get_reconstruction_latents(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Get bottleneck latents for reconstruction.

        Args:
            image: Input images, shape (B, C, H, W).

        Returns:
            Latents in 4D format, shape (B, C, H', W').
        """
        _, _, H, W = image.shape

        output = self.trunk(
            image,
            is_training=True,  # Must be True to get full dict output
            use_bottleneck=True,
        )

        patch_tokens = output['x_norm_patchtokens']  # (B, N, C)
        latents = self._patch_tokens_to_4d(patch_tokens, H, W)

        return latents

    def get_latents_decoded_images(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latents to reconstructed images.

        Args:
            latents: Latent features, shape (B, C, H', W').

        Returns:
            Reconstructed images, shape (B, 3, H, W).
        """
        if self.pixel_decoder is None:
            raise RuntimeError("Reconstruction not enabled. Set train_reconstruction=True in config.")

        return self.pixel_decoder(latents)

    def _patch_tokens_to_4d(
        self,
        patch_tokens: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """Convert patch tokens (B, N, C) to 4D format (B, C, H, W)."""
        B, N, C = patch_tokens.shape
        patch_size = self.config.vision_patch_size

        feat_h = img_h // patch_size
        feat_w = img_w // patch_size

        if N != feat_h * feat_w:
            raise ValueError(f"Patch count mismatch: {N} vs {feat_h * feat_w}")

        return patch_tokens.transpose(1, 2).reshape(B, C, feat_h, feat_w)

    # ==================== Forward ====================

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        forward_type: str = "clip",
    ) -> Dict[str, torch.Tensor]:
        """Unified forward with multiple modes.

        Args:
            image: Input images, shape (B, C, H, W).
            text: Input token IDs, shape (B, L).
            forward_type: One of:
                - "clip": Return CLIP features and logit_scale
                - "rec": Return reconstructed images
                - "feature": Return last layer features

        Returns:
            Dict with outputs based on forward_type.
        """
        if forward_type == "clip":
            return self._forward_clip(image, text)
        elif forward_type == "rec":
            return self._forward_reconstruction(image)
        elif forward_type == "feature":
            return self._forward_feature(image)
        else:
            raise ValueError(f"Invalid forward_type: {forward_type}")

    def _forward_clip(
        self,
        image: Optional[torch.Tensor],
        text: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward for CLIP mode."""
        result = {}

        if image is not None:
            result['image_features'] = self.get_clip_image_feature(image, normalize=True)

        if text is not None:
            result['text_features'] = self.get_clip_text_feature(text, normalize=True)

        result['logit_scale'] = self.logit_scale.exp()
        if self.logit_bias is not None:
            result['logit_bias'] = self.logit_bias

        return result

    def _forward_reconstruction(
        self,
        image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward for reconstruction mode."""
        if image is None:
            raise ValueError("image is required for reconstruction")

        latents = self.get_reconstruction_latents(image)
        reconstructed = self.get_latents_decoded_images(latents)

        return {
            'latents': latents,
            'reconstructed_image': reconstructed,
            'target_image': image,
        }

    def _forward_feature(
        self,
        image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward for feature extraction mode."""
        if image is None:
            raise ValueError("image is required for feature extraction")

        return self.get_last_layer_feature(image, use_bottleneck=True)
