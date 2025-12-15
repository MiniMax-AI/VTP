import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .vision_transformer import DinoVisionTransformer


class DinoVisionTransformerWithBottleneck(DinoVisionTransformer):
    """DINOv3 Vision Transformer with feature bottleneck for dimensionality reduction.

    This variant adds a linear projection layer to reduce the feature dimension
    after the transformer blocks, which is useful for tokenization tasks where
    a lower-dimensional representation is preferred.
    """

    def __init__(self, *args, vit_feature_bottleneck: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)

        # Bottleneck setup
        self.original_embed_dim = self.embed_dim
        if vit_feature_bottleneck is None:
            vit_feature_bottleneck = self.embed_dim

        self.vit_feature_bottleneck = vit_feature_bottleneck

        if self.vit_feature_bottleneck != self.original_embed_dim:
            self.feature_bottleneck = nn.Linear(self.original_embed_dim, self.vit_feature_bottleneck, bias=False)
            self.num_features = self.vit_feature_bottleneck
            logging.info(f"Feature bottleneck: {self.original_embed_dim} -> {self.vit_feature_bottleneck}")
        else:
            self.feature_bottleneck = None
            logging.info(f"No feature bottleneck needed: using original embed_dim={self.original_embed_dim}")

        # use init weights from official DINOv3
        self.init_weights()

    def init_weights(self):
        super().init_weights()

    def _apply_feature_bottleneck(self, features):
        if self.feature_bottleneck is not None:
            return self.feature_bottleneck(features)
        return features

    def forward_features(self, x: Union[torch.Tensor, List[torch.Tensor]], masks: Optional[torch.Tensor] = None, **kwargs):
        # Propagate drop_ratio down to parent
        output = super().forward_features(x, masks, drop_ratio=kwargs.get('drop_ratio', None))

        use_bottleneck = kwargs.get('use_bottleneck', True)

        if self.feature_bottleneck is None or not use_bottleneck:
            return output

        if isinstance(output, list):
            processed_output = []
            for item in output:
                processed_item = self._process_output_dict(item)
                processed_output.append(processed_item)
            return processed_output
        else:
            return self._process_output_dict(output)

    def _process_output_dict(self, output_dict):
        # Apply bottleneck to cls and patch tokens
        cls_token = self._apply_feature_bottleneck(output_dict["x_norm_clstoken"])

        patch_tokens = output_dict["x_norm_patchtokens"]

        batch_size, num_patches, _ = patch_tokens.shape
        patch_tokens = self.feature_bottleneck(patch_tokens.reshape(-1, self.original_embed_dim))
        patch_tokens = patch_tokens.reshape(batch_size, num_patches, self.vit_feature_bottleneck)

        output_dict["x_norm_clstoken"] = cls_token
        output_dict["x_norm_patchtokens"] = patch_tokens

        return output_dict

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        """
        Retrieves intermediate layer features from the model, bypassing the feature bottleneck.
        This method directly calls the parent class's implementation to return the
        original, high-dimensional features from the transformer blocks, which is
        typically desired for linear probing evaluations.
        """
        return super().get_intermediate_layers(
            x, n=n, reshape=reshape, return_class_token=return_class_token, norm=norm
        )

    def encode(self, x):
        """Encode images to latent features."""
        output = self.forward_features(x)
        b, n, c = output['x_norm_patchtokens'].shape
        h, w = x.shape[-2] // 16, x.shape[-1] // 16
        return output['x_norm_patchtokens'].reshape(b, h, w, c).permute(0, 3, 1, 2)
