from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ModelParams:
    n_layers: int = 2
    n_hidden: int = 800
    z_dim: int = 200
    beta: float = 0.1  # KL divergence term hyperparam for MVAE
    dropout: float = 0.1
    z_dropout: float = 0.3
    encode_covariates: bool = False
    use_cuda: bool = True


class FullyConnectedLayers(nn.Module):
    """
    Adapted from scvi.nn.FCLayers:
    https://docs.scvi-tools.org/en/stable/api/reference/scvi.nn.FCLayers.html
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        bias: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        """
        n_in: int - dimensionality of input
        n_out: int - dimensionality of output
        n_layers: int - number of fully-connected hidden layers
        n_hidden: int - number of nodes per hidden layer
        dropout_rate: float - dropout rate to apply to each of the hidden layers
        """
        super().__init__()
        dims = [n_in] + [n_hidden] * (n_layers - 1) + [n_out]
        layers = []
        for layer_in, layer_out in zip(dims, dims[1:]):
            layer = []
            layer.append(nn.Linear(layer_in, layer_out, bias=bias))
            if use_batch_norm:
                layer.append(nn.BatchNorm1d(layer_out, momentum=0.01, eps=0.001))
            if use_layer_norm:
                layers.append(nn.LayerNorm(layer_out, elementwise_affine=False))
            layer.append(activation_fn())
            layer.append(nn.Dropout(dropout_rate))
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                if x.dim() == 3:
                    x = torch.cat(
                        [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                    )
            x = layer(x)
            assert not torch.isnan(x).any(), f"NaNs in layer: {layer}"
        return x


class SamplingLayers(nn.Module):
    def __init__(self, n_in, n_out, dropout_rate) -> None:
        super(SamplingLayers, self).__init__()
        self.mean = nn.Sequential(nn.Linear(n_in, n_out), nn.Dropout(dropout_rate))
        self.logvar = nn.Sequential(nn.Linear(n_in, n_out), nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.mean(x), self.logvar(x)
