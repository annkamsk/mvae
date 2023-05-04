from typing import Tuple

from src.model import FullyConnectedLayers, ModelParams, SamplingLayers

from src.constants import BATCH_KEY, BATCH_N_KEY

from src.mvae.types import (
    Modality,
    ModalityInputT,
    ModalityOutput,
    ModalityOutputT,
    ModelInputT,
    ModelOutputT,
)

from src.latent import Latent, prior_expert, sample_latent
import torch
from torch import nn
import torch.nn.functional as F

from mudata import MuData


class ModalityLayers(nn.Module):
    """
    Architecture:
    modality -> private z
             -> batch1 z * batch2 z * ... * batchN z
             -> shared z
    """

    def __init__(self, n_in, n_batch, params: ModelParams) -> None:
        super(ModalityLayers, self).__init__()
        self.n_batch = n_batch
        self.params = params
        self.shared_sampling = SamplingLayers(
            params.n_hidden, params.z_dim, params.z_dropout
        )
        self.batch_sampling = SamplingLayers(
            params.n_hidden, params.z_dim * n_batch, params.z_dropout
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
        )
        self.decoder = FullyConnectedLayers(
            n_in=params.z_dim,
            n_out=params.n_hidden,
            n_layers=params.n_layers,
            n_hidden=params.n_hidden,
            dropout_rate=params.dropout,
            activation_fn=torch.nn.ReLU,
        )
        self.final = nn.Sequential(
            torch.nn.Linear(params.n_hidden, n_in), torch.nn.ReLU()
        )

    def forward(self, input: ModalityInputT) -> ModalityOutputT:
        latent_p, latent_mod, latent_s = self.encode(input)

        X = self.decode(latent_p.z + latent_mod.z)
        return dict(
            x=X,
            latent_p=latent_p.to_dict(),
            latent_mod=latent_mod.to_dict(),
            latent_s=latent_s.to_dict(),
        )

    def encode(
        self,
        input: ModalityInputT,
    ) -> Tuple[Latent, Latent, Latent]:
        """
        Encode data in latent space (Inference step).
        """
        batch_id = input["batch_id"]
        X = torch.squeeze(input["x"])
        y = self.encoder(X)

        return self.sample_latent(y, batch_id)

    def sample_latent(self, y, batch_id) -> Tuple[Latent, Latent, Latent]:
        mu_s, logvar_s = self.shared_sampling(y)
        mu_p_mod, logvar_p_mod = self.private_sampling(y)
        mu_p, logvar_p = self.batch_sampling(y)

        batch_encoding = torch.squeeze(
            F.one_hot(batch_id.to(torch.int64), num_classes=self.n_batch)
        ).unsqueeze(1)
        if self.n_batch > 1:
            mu_p = (
                mu_p.reshape((-1, self.params.z_dim, self.n_batch)) * batch_encoding
            ).sum(-1)
            logvar_p = (
                logvar_p.reshape((-1, self.params.z_dim, self.n_batch)) * batch_encoding
            ).sum(-1)
        else:
            mu_p = torch.zeros_like(mu_s).to(
                torch.device("cuda" if self.params.use_cuda else "cpu")
            )
            logvar_p = torch.zeros_like(logvar_s).to(
                torch.device("cuda" if self.params.use_cuda else "cpu")
            )

        z_p = sample_latent(mu_p, logvar_p)
        z_p_mod = sample_latent(mu_p_mod, logvar_p_mod)
        z_s = sample_latent(mu_s, logvar_s)
        return (
            Latent(z_p, mu_p, logvar_p),
            Latent(z_p_mod, mu_p_mod, logvar_p_mod),
            Latent(z_s, mu_s, logvar_s),
        )

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        X = self.decoder(z)
        return self.final(X)


class PoE(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1.0 / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1.0 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class MVAE(torch.nn.Module):
    """
    Architecture:
    RNA -> private z
        -> batch1 z * batch2 z * ... * batchN z
        -> shared z
    MSI -> private z
        -> batch1 z * batch2 z * ... x batchN z
        -> shared z
    RNA shared z * MSI shared z -> Z POE
    Z POE + RNA private z + RNA batch z product -> X1
    Z POE + MSI private z + MSI batch z product -> X2
    """

    def __init__(self, mdata: MuData, params: ModelParams = ModelParams(), **kwargs):
        super(MVAE, self).__init__()
        self.params = params
        self.device = "cuda" if params.use_cuda else "cpu"

        self.n_batch = mdata.uns[BATCH_N_KEY]

        rna_shape = mdata.mod[Modality.rna.name].shape
        msi_shape = mdata.mod[Modality.msi.name].shape

        self.rna = ModalityLayers(rna_shape[1], self.n_batch, params)
        self.msi = ModalityLayers(msi_shape[1], self.n_batch, params)
        self.poe = PoE()

    def forward(self, input: ModelInputT) -> ModelOutputT:
        rna_output = ModalityOutput.from_dict(
            self.rna(
                ModalityInputT(
                    x=input["rna"],
                    batch_id=input["batch_id"],
                )
            )
        )
        msi_output = ModalityOutput.from_dict(
            self.msi(
                ModalityInputT(
                    x=input["msi"],
                    batch_id=input["batch_id"],
                )
            )
        )
        poe = self.encode_poe(
            (1, input[BATCH_KEY].shape[0], self.params.z_dim),
            rna_output.latent_s,
            msi_output.latent_s,
        )
        rna_poe, rna_batch_free, msi_poe, msi_batch_free = self.decode_poe(
            poe,
            rna_output,
            msi_output,
        )

        # Translation losses
        msi_rna_loss = self.msi.decode(
            msi_output.latent_p.z + rna_output.latent_s.z,
        )
        rna_msi_loss = self.rna.decode(
            rna_output.latent_p.z + msi_output.latent_s.z,
        )

        return ModelOutputT(
            rna=rna_output.to_dict(),
            msi=msi_output.to_dict(),
            poe_latent=poe.to_dict(),
            rna_poe=rna_poe,
            rna_batch_free=rna_batch_free,
            msi_poe=msi_poe,
            msi_batch_free=msi_batch_free,
            rna_msi_loss=rna_msi_loss,
            msi_rna_loss=msi_rna_loss,
        )

    def encode_poe(
        self, size: Tuple[int], rna_latent_s: Latent, msi_latent_s: Latent
    ) -> Latent:
        mu, logvar = prior_expert(size, use_cuda=self.params.use_cuda)
        mu = torch.cat(
            (mu, rna_latent_s.mu.unsqueeze(0), msi_latent_s.mu.unsqueeze(0)),
            dim=0,
        )
        logvar = torch.cat(
            (
                logvar,
                rna_latent_s.logvar.unsqueeze(0),
                msi_latent_s.logvar.unsqueeze(0),
            ),
            dim=0,
        )
        mu, logvar = self.poe(mu, logvar)
        z_poe = sample_latent(mu, logvar, self.params.use_cuda)
        return Latent(z_poe, mu, logvar)

    def decode_poe(
        self,
        latent_poe: Latent,
        rna_output: ModalityOutput,
        msi_output: ModalityOutput,
    ):
        rna_poe = self.rna.decode(rna_output.latent_p.z + latent_poe.z)
        rna_batch_free = self.rna.decode(latent_poe.z + rna_output.latent_mod.z)

        msi_poe = self.msi.decode(msi_output.latent_p.z + latent_poe.z)
        msi_batch_free = self.msi.decode(latent_poe.z + msi_output.latent_mod.z)
        return rna_poe, rna_batch_free, msi_poe, msi_batch_free
