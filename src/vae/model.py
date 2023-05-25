from typing import Tuple
from src.constants import BATCH_N_KEY
from src.vae.types import VAEInputT, VAEOutputT
from src.latent import Latent, sample_latent
import torch
import torch.nn.functional as F
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


class VAEB(nn.Module):
    def __init__(self, anndata: AnnData, params: ModelParams) -> None:
        super(VAEB, self).__init__()
        self.params = params
        self.device = "cuda" if params.use_cuda else "cpu"

        n_in = anndata.shape[1]
        self.n_batch = anndata.uns[BATCH_N_KEY]

        self.batch_sampling = SamplingLayers(
            params.n_hidden, params.z_dim * self.n_batch, params.z_dropout
        )
        self.private_sampling = SamplingLayers(
            params.n_hidden, params.z_dim, params.z_dropout
        )
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
        latent_b, latent_mod = self.encode(input)

        X = self.decode(latent_b.z + latent_mod.z)
        return dict(
            x=X,
            latent_b=latent_b.to_dict(),
            latent_mod=latent_mod.to_dict(),
        )

    def encode(self, input: VAEInputT) -> Latent:
        batch_id = input["batch_id"]
        X = torch.squeeze(input["x"])
        y = self.encoder(X)
        return self.sample_latent(y, batch_id)

    def sample_latent(self, y, batch_id) -> Tuple[Latent, Latent]:
        mu_mod, logvar_mod = self.private_sampling(y)
        mu_b, logvar_b = self.batch_sampling(y)

        batch_encoding = torch.squeeze(
            F.one_hot(batch_id.to(torch.int64), num_classes=self.n_batch)
        ).unsqueeze(1)
        if self.n_batch > 1:
            mu_b = (
                mu_b.reshape((-1, self.params.z_dim, self.n_batch)) * batch_encoding
            ).sum(-1)
            logvar_b = (
                logvar_b.reshape((-1, self.params.z_dim, self.n_batch)) * batch_encoding
            ).sum(-1)
        else:
            mu_b = torch.zeros_like(mu_mod).to(
                torch.device("cuda" if self.params.use_cuda else "cpu")
            )
            logvar_b = torch.zeros_like(logvar_b).to(
                torch.device("cuda" if self.params.use_cuda else "cpu")
            )

        z_b = sample_latent(mu_b, logvar_b)
        z_mod = sample_latent(mu_mod, logvar_mod)
        return (
            Latent(z_b, mu_b, logvar_b),
            Latent(z_mod, mu_mod, logvar_mod),
        )

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        X = self.decoder(z)
        return self.final(X)
