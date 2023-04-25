from typing import Tuple

from src.smvae.types import (
    BATCH_KEY,
    ModalityInputT,
    ModalityOutputT,
    ModelInputT,
    ModelOutputT,
    ModalityOutput,
)
from src.types import Modality

from src.model import FullyConnectedLayers, ModelParams, SamplingLayers

from src.latent import Latent, initialize_latent, prior_expert, sample_latent
import torch
from torch import nn

from mudata import MuData


class ModalityLayers(nn.Module):
    """
    Architecture:
    modality -> private z
             -> shared z
    """

    def __init__(self, n_in, params: ModelParams) -> None:
        super(ModalityLayers, self).__init__()
        self.params = params
        self.shared_sampling = SamplingLayers(
            params.n_hidden, params.z_dim, params.z_dropout
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
        latent_p, latent_s = self.encode(input)

        X = self.decode(latent_p.z)
        return dict(
            x=X,
            latent_p=latent_p.to_dict(),
            latent_s=latent_s.to_dict(),
        )

    def encode(
        self,
        input: ModalityInputT,
    ) -> Tuple[Latent, Latent, Latent]:
        """
        Encode data in latent space (Inference step).
        """
        X = torch.squeeze(input["x"])
        y = self.encoder(X)

        return self.sample_latent(y)

    def sample_latent(self, y) -> Tuple[Latent, Latent]:
        mu_s, logvar_s = self.shared_sampling(y)
        mu_p, logvar_p = self.private_sampling(y)

        z_p = sample_latent(mu_p, logvar_p)
        z_s = sample_latent(mu_s, logvar_s)
        return (
            Latent(z_p, mu_p, logvar_p),
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


class SMVAE(torch.nn.Module):
    """
    Architecture:
    mod1 -> private z
        -> shared z
    mod2 -> private z
        -> shared z
    mod1 shared z * mod2 shared z -> Z POE
    Z POE + mod1 private z -> X1
    Z POE + mod2 private z -> X2
    """

    def __init__(
        self,
        mdata: MuData,
        params: ModelParams = ModelParams(),
        use_cuda=True,
    ):
        super(SMVAE, self).__init__()
        self.params = params
        self.device = "cuda" if use_cuda else "cpu"

        rna_shape = mdata.mod[Modality.rna.name].shape
        msi_shape = mdata.mod[Modality.msi.name].shape

        print(rna_shape)
        print(msi_shape)

        self.mod1 = ModalityLayers(rna_shape[1], params)
        self.mod2 = ModalityLayers(msi_shape[1], params)
        self.poe = PoE()

    def forward(self, input: ModelInputT) -> ModelOutputT:
        mod1_output = ModalityOutput.from_dict(
            self.mod1(
                ModalityInputT(
                    x=input[Modality.rna.name],
                    batch_id=input[BATCH_KEY],
                )
            )
        )
        mod2_output = ModalityOutput.from_dict(
            self.mod2(
                ModalityInputT(
                    x=input[Modality.msi.name],
                    batch_id=input[BATCH_KEY],
                )
            )
        )
        poe = self.encode_poe(
            (1, input[BATCH_KEY].shape[0], self.params.z_dim),
            mod1_output.latent_s,
            mod2_output.latent_s,
        )
        mod1_poe, mod2_poe = self.decode_poe(
            poe,
            mod1_output,
            mod2_output,
        )

        # Translation losses
        mod1_mod2_loss = self.mod2.decode(
            mod2_output.latent_p.z + mod1_output.latent_s.z,
        )
        mod2_mod1_loss = self.mod1.decode(
            mod1_output.latent_p.z + mod2_output.latent_s.z,
        )

        return ModelOutputT(
            rna=mod1_output.to_dict(),
            msi=mod2_output.to_dict(),
            poe_latent=poe.to_dict(),
            mod1_poe=mod1_poe,
            mod2_poe=mod2_poe,
            mod1_mod2_loss=mod1_mod2_loss,
            mod2_mod1_loss=mod2_mod1_loss,
        )

    def encode_poe(
        self, size: Tuple[int], mod1_latent_s: Latent, mod2_latent_s: Latent
    ) -> Latent:
        mu, logvar = prior_expert(size, use_cuda=self.params.use_cuda)
        mu = torch.cat(
            (mu, mod1_latent_s.mu.unsqueeze(0), mod2_latent_s.mu.unsqueeze(0)),
            dim=0,
        )
        logvar = torch.cat(
            (
                logvar,
                mod1_latent_s.logvar.unsqueeze(0),
                mod2_latent_s.logvar.unsqueeze(0),
            ),
            dim=0,
        )
        mu, logvar = self.poe(mu, logvar)
        z_poe = sample_latent(mu, logvar, self.params.use_cuda)
        return Latent(z_poe, mu, logvar)

    def decode_poe(
        self,
        latent_poe: Latent,
        mod1_output: ModalityOutput,
        mod2_output: ModalityOutput,
    ):
        mod1_poe = self.mod1.decode(mod1_output.latent_p.z + latent_poe.z)
        mod2_poe = self.mod2.decode(mod2_output.latent_p.z + latent_poe.z)
        return mod1_poe, mod2_poe
