from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from src.types import (
    Modality,
    ModalityInputT,
    ModalityOutput,
    ModalityOutputT,
    ModelInputT,
    ModelOutputT,
)

from src.latent import Latent, initialize_latent, sample_latent
import torch
from torch import nn
import torch.nn.functional as F

from mudata import MuData


@dataclass
class MVAEParams:
    n_layers: int = 2
    n_hidden: int = 800
    z_dim: int = 200
    beta: float = 0.1  # KL divergence term hyperparam for MVAE
    dropout: float = 0.1
    z_dropout: float = 0.3
    encode_covariates: bool = False


class FullyConnectedLayers(nn.Sequential):
    def __init__(
        self,
        n_in,
        n_out,
        n_layers=1,
        n_hidden=128,
        dropout_rate=0.1,
        activation_fn=torch.nn.ReLU,
    ):
        """
        n_in: int - dimensionality of input
        n_out: int - dimensionality of output
        n_layers: int - number of fully-connected hidden layers
        n_hidden: int - number of nodes per hidden layer
        dropout_rate: float - dropout rate to apply to each of the hidden layers
        """
        dims = [n_in] + [n_hidden] * n_layers + [n_out]
        layers = []
        for layer_in, layer_out in zip(dims, dims[1:]):
            layers.append(nn.Linear(layer_in, layer_out))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
        super().__init__(*layers)


class SamplingLayers(nn.Module):
    def __init__(self, n_in, n_out, dropout_rate) -> None:
        super(SamplingLayers, self).__init__()
        self.mean = nn.Sequential(nn.Linear(n_in, n_out), nn.Dropout(dropout_rate))
        self.logvar = nn.Sequential(nn.Linear(n_in, n_out), nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.mean(x), self.logvar(x)


class ModalityLayers(nn.Module):
    """
    Architecture:
    modality -> private z
             -> batch1 z * batch2 z * ... * batchN z
             -> shared z
    """

    def __init__(self, n_in, n_batch, params: MVAEParams) -> None:
        super(ModalityLayers, self).__init__()
        self.n_batch = n_batch
        self.params = params
        self.shared_sampling = SamplingLayers(
            params.n_hidden, params.z_dim, params.dropout
        )
        self.batch_sampling = SamplingLayers(
            params.n_hidden, params.z_dim * n_batch, params.dropout
        )
        self.private_sampling = SamplingLayers(
            params.n_hidden, params.z_dim, params.dropout
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

    def forward(self, input: ModalityInputT) -> Dict:
        latent_p, latent_mod, latent_s = self.encode(input)

        batch_only = self.decode(latent_p.z, input["batch_id"], input["cat_covs"])
        X = self.decode(latent_p.z + latent_mod.z, input["batch_id"], input["cat_covs"])
        return dict(
            X=X,
            latent_p=asdict(latent_p),
            latent_mod=asdict(latent_mod),
            latent_s=asdict(latent_s),
            batch_only=asdict(batch_only),
        )

    def encode(
        self,
        input: ModalityInputT,
    ) -> Tuple[Latent, Latent, Latent]:
        """
        Encode data in latent space (Inference step).
        """
        if input["extra_categorical_covs"] and self.params.encode_covariates:
            categorical_input = torch.split(input["extra_categorical_covs"], 1, dim=1)
        else:
            categorical_input = tuple()

        batch_id = input["batch_id"][input["idxs"], :]
        X = torch.squeeze(input["x"][input["idxs"], :, :])
        y = self.encoder(X, batch_id, *categorical_input)

        latent_size = [input["idxs"].shape[0], self.params.z_dim]
        latent_p = initialize_latent(latent_size, use_cuda=self.params.use_cuda)
        latent_mod = initialize_latent(latent_size, use_cuda=self.params.use_cuda)
        latent_s = initialize_latent(latent_size, use_cuda=self.params.use_cuda)

        post_latent = self.sample_latent(y, batch_id)
        for prior, post in zip([latent_p, latent_mod, latent_s], post_latent):
            prior.update(post, input["idxs"])
        return [latent_p, latent_mod, latent_s]

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
            mu_p = torch.zeros_like(mu_s).to(self.params.device)
            logvar_p = torch.zeros_like(logvar_s).to(self.params.device)

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
        batch_id: torch.Tensor,
        cat_covs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        X = self.decoder(z, batch_id, *categorical_input)
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

    def __init__(
        self, mdata: MuData, params: MVAEParams = MVAEParams(), use_cuda=True, **kwargs
    ):
        super(MVAE, self).__init__()
        self.params = params
        self.device = "cuda" if use_cuda else "cpu"

        self.n_batch_mod1 = mdata.mod[Modality.rna.name].uns["n_batch"]
        self.n_batch_mod2 = mdata.mod[Modality.msi.name].uns["n_batch"]

        print("N batches for mod1: ", str(self.n_batch_mod1))
        print("N batches for mod2: ", str(self.n_batch_mod2))

        rna_shape = mdata.mod[Modality.rna.name].shape
        msi_shape = mdata.mod[Modality.msi.name].shape

        print(rna_shape)
        print(msi_shape)

        self.rna = ModalityLayers(rna_shape[1], self.n_batch_mod1, params)
        self.msi = ModalityLayers(msi_shape[1], self.n_batch_mod2, params)
        self.poe = PoE()

    def forward(self, input: ModelInputT) -> ModelOutputT:
        mod_id = input["mod_id"]

        mod_idxs_1 = (mod_id == 1) | (mod_id == 3)
        mod_idxs_2 = (mod_id == 2) | (mod_id == 3)

        rna_output = ModalityOutput.from_dict(
            self.rna(
                ModalityInputT(
                    x=input["rna"],
                    mod_id=mod_idxs_1,
                    batch_id=input["batch_id1"],
                    cat_covs=input["extra_categorical_covs"],
                )
            )
        )
        msi_output = ModalityOutput.from_dict(
            self.msi(
                ModalityInputT(
                    x=input["msi"],
                    mod_id=mod_idxs_2,
                    batch_id=input["batch_id2"],
                    cat_covs=input["extra_categorical_covs"],
                )
            )
        )
        poe = self.encode_poe(
            [1, mod_id.shape[0], self.params.z_dim],
            rna_output.latent_s,
            msi_output.latent_s,
        )
        rna_poe, rna_batch_free, msi_poe, msi_batch_free = self.decode_poe(
            poe,
            rna_output,
            msi_output,
            input["batch_id1"],
            input["batch_id2"],
            input["extra_categorical_covs"],
        )

        # Translation losses
        rna_msi_loss = self.msi.decode(
            rna_output.latent_p.z + msi_output.latent_s.z,
            input["batch_id2"],
            input["extra_categorical_covs"],
        )
        msi_rna_loss = self.rna.decode(
            msi_output.latent_p.z + rna_output.latent_s.z,
            input["batch_id1"],
            input["extra_categorical_covs"],
        )
        return ModelOutputT(
            rna=asdict(rna_output),
            msi=asdict(msi_output),
            poe=asdict(poe),
            rna_poe=rna_poe,
            rna_batch_free=rna_batch_free,
            msi_poe=msi_poe,
            msi_batch_free=msi_batch_free,
            rna_msi_loss=rna_msi_loss,
            msi_rna_loss=msi_rna_loss,
        )

    def encode_poe(
        self, size: List[int], rna_latent_s: Latent, msi_latent_s: Latent
    ) -> Latent:
        mu, logvar = initialize_latent(size, use_cuda=self.params.use_cuda)
        mu = torch.cat(
            (mu, rna_latent_s.mu.unsqueeze(0), msi_latent_s.mu.unsqueeze(0)), dim=0
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
        batch_id1,
        batch_id2,
        cat_covs,
    ):
        rna_poe = self.rna.decode(
            rna_output.latent_p.z + latent_poe.z, batch_id1, cat_covs
        )
        rna_batch_free = self.rna.decode(
            latent_poe.z + rna_output.latent_mod.z, batch_id1, cat_covs
        )

        msi_poe = self.msi.decode(
            msi_output.latent_p.z + latent_poe.z, batch_id2, cat_covs
        )
        msi_batch_only = self.msi.decode(
            latent_poe.z + msi_output.latent_mod.z, batch_id2, cat_covs
        )
        return rna_poe, rna_batch_free, msi_poe, msi_batch_only
