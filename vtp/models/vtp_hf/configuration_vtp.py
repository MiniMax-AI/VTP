"""VTP model configuration for HuggingFace compatibility."""

from typing import Optional
from transformers import PretrainedConfig


class VTPConfig(PretrainedConfig):
    """Configuration class for VTP (Visual Tokenizer Pre-training) model.

    This configuration supports two main modes:
    - CLIP-style contrastive learning (train_clip)
    - Image reconstruction (train_reconstruction)

    Args:
        image_size: Input image size.
        train_clip: Whether to enable CLIP-style contrastive learning.
        train_reconstruction: Whether to enable image reconstruction.

        Vision Encoder (DinoVisionTransformerWithBottleneck):
            vision_patch_size: Patch size for vision transformer.
            vision_embed_dim: Embedding dimension for vision transformer.
            vision_depth: Number of transformer blocks.
            vision_num_heads: Number of attention heads.
            vision_mlp_ratio: MLP hidden dim ratio.
            vision_ffn_layer: FFN layer type ('mlp', 'swiglu', etc.).
            vision_norm_layer: Normalization layer type ('layernorm', 'rmsnorm').
            vision_init_values: LayerScale init value (None to disable).
            vision_use_qk_norm: Whether to use QK normalization.
            vision_feature_bottleneck: Bottleneck dimension for features.
            vision_bottleneck_ae_only: Use bottleneck only for reconstruction.
            vision_clip_feat: Feature type for CLIP ('cls' or 'pooled').

        Text Encoder (TextTransformer):
            text_context_length: Maximum text sequence length.
            text_vocab_size: Vocabulary size.
            text_embed_dim: Text embedding dimension.
            text_num_heads: Number of attention heads.
            text_depth: Number of transformer layers.
            text_mlp_ratio: MLP hidden dim ratio.
            text_ls_init_value: LayerScale init value.
            text_embed_cls: Whether to add CLS embedding.
            text_pad_id: Padding token ID.
            text_no_causal_mask: Disable causal masking.
            text_pool_type: Pooling type ('argmax', 'last', 'first').
            text_proj_type: Projection type ('linear', 'mlp', None).
            text_proj_bias: Whether projection has bias.
            text_output_tokens: Whether to output all tokens.
            text_quick_gelu: Use QuickGELU activation.

        Pixel Decoder (DinoV3PixelDecoder):
            decoder_embed_dim: Decoder embedding dimension.
            decoder_num_heads: Number of attention heads.
            decoder_depth: Number of decoder layers.
            decoder_ffn_layer: FFN layer type.
            decoder_norm_layer: Normalization layer type.
            decoder_init_values: LayerScale init value.
            decoder_use_qk_norm: Whether to use QK normalization.

        Runtime:
            init_logit_scale: Initial logit scale for CLIP.
            init_logit_bias: Initial logit bias for SigLIP.
            nonscalar_logit_scale: Use non-scalar logit scale.
    """

    model_type = "vtp"

    def __init__(
        self,
        # General
        image_size: int = 256,
        train_clip: bool = True,
        train_reconstruction: bool = True,
        # Vision Encoder
        vision_patch_size: int = 16,
        vision_embed_dim: int = 768,
        vision_depth: int = 12,
        vision_num_heads: int = 12,
        vision_mlp_ratio: float = 4.0,
        vision_ffn_layer: str = "swiglu",
        vision_norm_layer: str = "rmsnorm",
        vision_init_values: Optional[float] = None,
        vision_use_qk_norm: bool = False,
        vision_feature_bottleneck: int = 64,
        vision_bottleneck_ae_only: bool = True,
        vision_clip_feat: str = "cls",
        # Text Encoder
        text_context_length: int = 77,
        text_vocab_size: int = 49408,
        text_embed_dim: int = 768,
        text_num_heads: int = 12,
        text_depth: int = 12,
        text_mlp_ratio: float = 4.0,
        text_ls_init_value: Optional[float] = None,
        text_embed_cls: bool = False,
        text_pad_id: int = 0,
        text_no_causal_mask: bool = False,
        text_pool_type: str = "argmax",
        text_proj_type: str = "linear",
        text_proj_bias: bool = False,
        text_output_tokens: bool = False,
        text_quick_gelu: bool = False,
        # Pixel Decoder
        decoder_embed_dim: int = 768,
        decoder_num_heads: int = 12,
        decoder_depth: int = 12,
        decoder_ffn_layer: str = "swiglu",
        decoder_norm_layer: str = "layernorm",
        decoder_init_values: Optional[float] = None,
        decoder_use_qk_norm: bool = False,
        # Runtime
        init_logit_scale: Optional[float] = None,
        init_logit_bias: Optional[float] = None,
        nonscalar_logit_scale: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # General
        self.image_size = image_size
        self.train_clip = train_clip
        self.train_reconstruction = train_reconstruction

        # Vision Encoder
        self.vision_patch_size = vision_patch_size
        self.vision_embed_dim = vision_embed_dim
        self.vision_depth = vision_depth
        self.vision_num_heads = vision_num_heads
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_ffn_layer = vision_ffn_layer
        self.vision_norm_layer = vision_norm_layer
        self.vision_init_values = vision_init_values
        self.vision_use_qk_norm = vision_use_qk_norm
        self.vision_feature_bottleneck = vision_feature_bottleneck
        self.vision_bottleneck_ae_only = vision_bottleneck_ae_only
        self.vision_clip_feat = vision_clip_feat

        # Text Encoder
        self.text_context_length = text_context_length
        self.text_vocab_size = text_vocab_size
        self.text_embed_dim = text_embed_dim
        self.text_num_heads = text_num_heads
        self.text_depth = text_depth
        self.text_mlp_ratio = text_mlp_ratio
        self.text_ls_init_value = text_ls_init_value
        self.text_embed_cls = text_embed_cls
        self.text_pad_id = text_pad_id
        self.text_no_causal_mask = text_no_causal_mask
        self.text_pool_type = text_pool_type
        self.text_proj_type = text_proj_type
        self.text_proj_bias = text_proj_bias
        self.text_output_tokens = text_output_tokens
        self.text_quick_gelu = text_quick_gelu

        # Pixel Decoder
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_heads = decoder_num_heads
        self.decoder_depth = decoder_depth
        self.decoder_ffn_layer = decoder_ffn_layer
        self.decoder_norm_layer = decoder_norm_layer
        self.decoder_init_values = decoder_init_values
        self.decoder_use_qk_norm = decoder_use_qk_norm

        # Runtime
        self.init_logit_scale = init_logit_scale
        self.init_logit_bias = init_logit_bias
        self.nonscalar_logit_scale = nonscalar_logit_scale

    @classmethod
    def from_vtp_yaml(cls, yaml_path: str) -> "VTPConfig":
        """Create VTPConfig from legacy VTP YAML configuration file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            VTPConfig instance.
        """
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(yaml_path)

        vision_cfg = cfg.vtp_model.vision_encoder
        text_cfg = cfg.vtp_model.text_encoder
        decoder_cfg = cfg.vtp_model.pixel_decoder
        training_cfg = cfg.training

        return cls(
            # General
            image_size=cfg.data.image_size,
            train_clip=training_cfg.train_clip,
            train_reconstruction=training_cfg.train_reconstruction,
            # Vision Encoder
            vision_patch_size=vision_cfg.patch_size,
            vision_embed_dim=vision_cfg.embed_dim,
            vision_depth=vision_cfg.depth,
            vision_num_heads=vision_cfg.num_heads,
            vision_mlp_ratio=vision_cfg.mlp_ratio,
            vision_ffn_layer=vision_cfg.ffn_layer,
            vision_norm_layer=vision_cfg.norm_type,
            vision_init_values=vision_cfg.get("init_values", None),
            vision_use_qk_norm=vision_cfg.get("use_qk_norm", False),
            vision_feature_bottleneck=vision_cfg.vit_feature_bottleneck,
            vision_bottleneck_ae_only=vision_cfg.bottleneck_ae_only,
            vision_clip_feat=vision_cfg.clip_feat,
            # Text Encoder
            text_context_length=text_cfg.context_length,
            text_vocab_size=text_cfg.vocab_size,
            text_embed_dim=text_cfg.embed_dim,
            text_num_heads=text_cfg.heads,
            text_depth=text_cfg.layers,
            text_mlp_ratio=text_cfg.mlp_ratio,
            text_ls_init_value=text_cfg.get("ls_init_value", None),
            text_embed_cls=text_cfg.embed_cls,
            text_pad_id=text_cfg.pad_id,
            text_no_causal_mask=text_cfg.no_causal_mask,
            text_pool_type=text_cfg.pool_type,
            text_proj_type=text_cfg.proj_type,
            text_proj_bias=text_cfg.proj_bias,
            text_output_tokens=text_cfg.output_tokens,
            text_quick_gelu=text_cfg.quick_gelu,
            # Pixel Decoder
            decoder_embed_dim=decoder_cfg.embed_dim,
            decoder_num_heads=decoder_cfg.num_heads,
            decoder_depth=decoder_cfg.depth,
            decoder_ffn_layer=decoder_cfg.ffn_layer,
            decoder_norm_layer=decoder_cfg.norm_layer,
            decoder_init_values=decoder_cfg.get("layerscale_init", None),
            decoder_use_qk_norm=decoder_cfg.get("use_qk_norm", False),
            # Runtime
            init_logit_scale=training_cfg.get("init_logit_scale", None),
            init_logit_bias=training_cfg.get("init_logit_bias", None),
            nonscalar_logit_scale=training_cfg.get("nonscalar_logit_scale", False),
        )
