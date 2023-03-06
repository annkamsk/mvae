from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from src.types import Modality, ModalityOutputT, ModelInputT, ModelOutputT

from src.latent import Latent, initialize_latent, sample_latent
import torch
from torch import nn
import torch.nn.functional as F

from mudata import MuData


@dataclass
class MVAEParams:
    n_layers = 2
    n_hidden = 800
    z_dim = 200
    beta = 0.1  # KL divergence term hyperparam for MVAE
    dropout = 0.1
    z_dropout = 0.3
    encode_covariates = False
    batch_size = 32


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


class SamplingLayers:
    def __init__(self, n_in, n_out, dropout_rate) -> None:
        self.mean = nn.Sequential(nn.Linear(n_in, n_out), nn.Dropout(dropout_rate))
        self.logvar = nn.Sequential(nn.Linear(n_in, n_out), nn.Dropout(dropout_rate))


class ModalityLayers:
    """
    Architecture:
    modality -> private z
             -> batch1 z * batch2 z * ... * batchN z
             -> shared z
    """

    def __init__(self, n_in, n_batch, params: MVAEParams) -> None:
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

    def initialize_latent(self, size: List[int]) -> Tuple[Latent, Latent, Latent]:
        """
        Initialize latent for all layers.
        """
        latent_p = initialize_latent(size, use_cuda=self.params.use_cuda)
        latent_mod = initialize_latent(size, use_cuda=self.params.use_cuda)
        latent_s = initialize_latent(size, use_cuda=self.params.use_cuda)
        return latent_p, latent_mod, latent_s

    def sample_latent(self, y, batch_id) -> Tuple[Latent, Latent, Latent]:
        mu_s, logvar_s = self.shared_sampling.mean(y), self.shared_sampling.logvar(y)
        mu_p_mod, logvar_p_mod = self.private_sampling.mean(
            y
        ), self.private_sampling.logvar(y)
        mu_p, logvar_p = self.batch_sampling.mean(y), self.batch_sampling.logvar(y)

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
            mu_p = torch.zeros_like(mu_s).to(self.device)
            logvar_p = torch.zeros_like(logvar_s).to(self.device)

        z_p = sample_latent(mu_p, logvar_p)
        z_p_mod = sample_latent(mu_p_mod, logvar_p_mod)
        z_s = sample_latent(mu_s, logvar_s)
        return (
            Latent(z_p, mu_p, logvar_p),
            Latent(z_p_mod, mu_p_mod, logvar_p_mod),
            Latent(z_s, mu_s, logvar_s),
        )


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

        self.n_batch_mod1 = mdata.mod[Modality.rna.name].uns["_scvi"]["summary_stats"][
            "n_batch"
        ]
        self.n_batch_mod2 = mdata.mod[Modality.msi.name].uns["_scvi"]["summary_stats"][
            "n_batch"
        ]

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

        # Encoding

        rna_latent_p, rna_latent_mod, rna_latent_s = self.encode_modality(
            self.rna,
            input["rna"],
            mod_idxs_1,
            input["batch_id1"],
            input["extra_categorical_covs"],
        )
        msi_latent_p, msi_latent_mod, msi_latent_s = self.encode_modality(
            self.msi,
            input["msi"],
            mod_idxs_2,
            input["batch_id2"],
            input["extra_categorical_covs"],
        )
        poe_latent = self.encode_poe(
            [1, mod_id.shape[0], self.params.z_dim], rna_latent_s, msi_latent_s
        )

        # Reconstruction

        rna_latent_decoded = self.decode_modality(
            self.rna,
            rna_latent_p,
            rna_latent_mod,
            poe_latent,
            input["batch_id1"],
            input["extra_categorical_covs"],
        )
        msi_latent_decoded = self.decode_modality(
            self.msi,
            msi_latent_p,
            msi_latent_mod,
            poe_latent,
            input["batch_id2"],
            input["extra_categorical_covs"],
        )

        # Translation losses
        rna_msi_loss = self.decode(
            self.msi,
            rna_latent_p + msi_latent_s,
            input["batch_id2"],
            input["extra_categorical_covs"],
        )
        msi_rna_loss = self.decode(
            self.rna,
            msi_latent_p + rna_latent_s,
            input["batch_id1"],
            input["extra_categorical_covs"],
        )
        return ModelOutputT(
            rna=ModalityOutputT(
                *rna_latent_decoded,
                asdict(rna_latent_p),
                asdict(rna_latent_mod),
                asdict(rna_latent_s),
            ),
            msi=ModalityOutputT(
                *msi_latent_decoded,
                asdict(msi_latent_p),
                asdict(msi_latent_mod),
                asdict(msi_latent_s),
            ),
            poe_latent=poe_latent,
            rna_msi_loss=rna_msi_loss,
            msi_rna_loss=msi_rna_loss,
        )

    def encode_modality(
        self,
        modality: ModalityLayers,
        data: torch.Tensor,
        batch_id: torch.Tensor,
        mod_idxs,
        cat_covs: Optional[torch.Tensor] = None,
    ) -> Tuple[Latent, Latent, Latent]:
        """
        Encode data in latent space (Inference step).
        Parameters
        ----------
        data - input data
        mod_idxs - observations where modality is present
        batch_index - batch information for samples
        cat_covs - categorical covariates
        """
        if cat_covs and self.params.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        batch_id = batch_id[mod_idxs, :]
        X = torch.squeeze(data[mod_idxs, :, :])
        y = modality.encoder(X, batch_id, *categorical_input)

        prior_latent = modality.initialize_latent(
            [mod_idxs.shape[0], self.params.z_dim]
        )
        post_latent = modality.sample_latent(y, batch_id)
        for prior, post in zip(prior_latent, post_latent):
            prior.update(post)
        return prior_latent

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

    def decode_modality(
        self,
        modality: ModalityLayers,
        latent_p: Latent,
        latent_mod: Latent,
        latent_poe: Latent,
        batch_id: torch.Tensor,
        cat_covs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_only = self.decode(modality, latent_p.z, batch_id, cat_covs)
        X_poe = self.decode(
            modality,
            latent_p.z + latent_poe.z,
            batch_id,
            cat_covs,
        )
        X = self.decode(
            modality,
            latent_p.z + latent_mod.z,
            batch_id,
            cat_covs,
        )
        X_batch_free = self.decode(
            modality,
            latent_poe.z + latent_mod.z,
            batch_id,
            cat_covs,
        )
        return X, batch_only, X_poe, X_batch_free

    def decode(self, modality_layers: ModalityLayers, z, batch_id, cat_covs=None):
        """
        Decode data from latent space.
        Parameters
        ----------
        z - data embedded in latent space
        batch_index - batch information for samples
        cat_covs - categorical covariates
        """
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        X = modality_layers.decoder(z, batch_id, *categorical_input)
        X = modality_layers.final(X)
        return X
