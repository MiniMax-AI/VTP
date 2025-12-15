import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


class DINOHead(nn.Module):
    """Unified DINO Head supporting both weight_norm and flexible forward modes.

    This head is used for self-supervised learning in DINO/iBOT style training.
    It projects features through an MLP and a final layer with optional weight normalization.

    Args:
        in_dim: Input feature dimension
        out_dim: Output dimension (typically number of prototypes)
        use_bn: Whether to use batch normalization in MLP
        nlayers: Number of MLP layers
        hidden_dim: Hidden dimension of MLP
        bottleneck_dim: Bottleneck dimension before final projection
        mlp_bias: Whether to use bias in MLP layers
        use_weight_norm: Whether to use weight normalization on last layer
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        mlp_bias: bool = True,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )

        self.use_weight_norm = use_weight_norm
        if use_weight_norm:
            self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
        else:
            self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

        self.apply(self._init_weights)

    def init_weights(self) -> None:
        """Reinitialize weights."""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        no_last_layer: bool = False,
        only_last_layer: bool = False,
    ) -> torch.Tensor:
        """Forward pass with flexible layer control.

        Args:
            x: Input tensor
            no_last_layer: If True, skip the final projection layer
            only_last_layer: If True, only apply the final projection layer

        Returns:
            Processed tensor
        """
        if not only_last_layer:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)

        if not no_last_layer:
            x = self.last_layer(x)

        return x


def _build_mlp(
    nlayers: int,
    in_dim: int,
    bottleneck_dim: int,
    hidden_dim: int = None,
    use_bn: bool = False,
    bias: bool = True,
) -> nn.Module:
    """Build MLP for DINO head.

    Args:
        nlayers: Number of layers
        in_dim: Input dimension
        bottleneck_dim: Output dimension
        hidden_dim: Hidden dimension
        use_bn: Whether to use batch normalization
        bias: Whether to use bias

    Returns:
        MLP module
    """
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
