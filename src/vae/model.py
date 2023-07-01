from src.vae.types import VAEInputT, VAEOutputT
from src.latent import Latent, sample_latent
import torch
from src.model import FullyConnectedLayers, ModelParams, SamplingLayers

import torch.nn as nn
from anndata import AnnData


class VAE(nn.Module):
    def __init__(self, anndata: AnnData, params: ModelParams) -> None:
        super(VAE, self).__init__()
        self.params = params
        self.device = "cuda" if params.use_cuda else "cpu"

        n_in = anndata.shape[1]

        self.sampling = SamplingLayers(params.n_hidden, params.z_dim, params.z_dropout)

        self.encoder = FullyConnectedLayers(
            n_in=n_in,
            n_out=params.n_hidden,
            n_layers=params.n_layers,
            n_hidden=params.n_hidden,
            dropout_rate=params.dropout,
            activation_fn=torch.nn.ReLU,
            use_batch_norm=True,
        )
        self.decoder = FullyConnectedLayers(
            n_in=params.z_dim,
            n_out=params.n_hidden,
            n_layers=params.n_layers,
            n_hidden=params.n_hidden,
            dropout_rate=params.dropout,
            activation_fn=torch.nn.ReLU,
            use_batch_norm=True,
        )
        self.final = nn.Sequential(
            torch.nn.Linear(params.n_hidden, n_in), torch.nn.ReLU()
        )

    def forward(self, input: VAEInputT) -> VAEOutputT:
        latent = self.encode(input)

        X = self.decode(latent.z)
        return dict(
            x=X,
            latent=latent.to_dict(),
        )

    def encode(self, input: VAEInputT) -> Latent:
        X = torch.squeeze(input["x"])
        y = self.encoder(X)
        assert not torch.isnan(y).any()

        mu, logvar = self.sampling(y)
        z = sample_latent(mu, logvar)
        return Latent(mu=mu, logvar=logvar, z=z)

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        X = self.decoder(z)
        return self.final(X)
