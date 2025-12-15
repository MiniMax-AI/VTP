"""VTP HuggingFace-compatible model.

Example usage:
    >>> from vtp.models.vtp_hf import VTPConfig, VTPModel
    >>>
    >>> # Create from config
    >>> config = VTPConfig(vision_embed_dim=768, train_clip=True)
    >>> model = VTPModel(config)
    >>>
    >>> # Load from pretrained
    >>> model = VTPModel.from_pretrained("path/to/model")
    >>>
    >>> # Create from legacy YAML
    >>> config = VTPConfig.from_vtp_yaml("configs/vtp_b.yaml")
    >>> model = VTPModel(config)
"""

from .configuration_vtp import VTPConfig
from .modeling_vtp import VTPModel, VTPPreTrainedModel

__all__ = [
    "VTPConfig",
    "VTPModel",
    "VTPPreTrainedModel",
]
