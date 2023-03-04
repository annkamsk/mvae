from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.latent import Latent, initialize_latent, sample_latent

from train import Loss
from dataloader import (
    Modality,
    MultimodalDatasetItem,
    MultimodalDatasetItemT,
    mudata_to_dataloader,
)
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import datetime

from tqdm import tqdm, trange
from scvi.nn import FCLayers

from scipy import sparse
import umap
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
from mudata import MuData

from .constants import _CONSTANTS
from .utils import (
    SplitMethod,
    split_into_train_test,
    _anndata_loader,
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


# Utils functions for mmd (maximum mean discrepancy)
def pairwise_distance(x, y):
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, alphas):
    """Computes multiscale-RBF kernel between x and y.
    Parameters
    ----------
    x: torch.Tensor
         Tensor with shape [batch_size, z_dim].
    y: torch.Tensor
         Tensor with shape [batch_size, z_dim].
    alphas: Tensor
    Returns
    -------
    Returns the computed multiscale-RBF kernel between x and y.
    """

    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1.0 / (2.0 * alphas)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def loss_mod(pred, real, mod, modality, dropout=False):
    if modality == _CONSTANTS.MODALITY1_KEY:
        pred = pred[(mod == 1) | (mod == 3)]
        real = real[(mod == 1) | (mod == 3)]

        if dropout:
            dropout_mask = (real != 0).float()
            n_nonzero_features = dropout_mask.sum().int()
            mask_size = dropout_mask.size()
            dropout_mask = dropout_mask.reshape(-1).clone()
            dropout_mask[torch.randperm(len(dropout_mask))[:n_nonzero_features]] = 1
            dropout_mask = dropout_mask.reshape(mask_size)
            if _CONSTANTS.MODALITY1_LOSS == "mse":
                return torch.sum((real - pred).pow(2) * dropout_mask) / torch.sum(
                    dropout_mask
                )
            elif _CONSTANTS.MODALITY1_LOSS == "bce":
                class_weight = (
                    dropout_mask.shape[0] * dropout_mask.shape[1] - n_nonzero_features
                ) / n_nonzero_features
                return torch.nn.BCEWithLogitsLoss(
                    reduce="mean", pos_weight=class_weight
                )(pred, real)
        else:
            if _CONSTANTS.MODALITY1_LOSS == "mse":
                return F.mse_loss(pred, real, reduction="mean")
            elif _CONSTANTS.MODALITY1_LOSS == "bce":
                return torch.nn.BCEWithLogitsLoss(reduce="mean")(pred, real)

    elif modality == _CONSTANTS.MODALITY2_KEY:
        pred = pred[(mod == 2) | (mod == 3)]
        real = real[(mod == 2) | (mod == 3)]
        if dropout:
            dropout_mask = (real != 0).float()
            n_nonzero_features = dropout_mask.sum().int()
            mask_size = dropout_mask.size()
            dropout_mask = dropout_mask.reshape(-1).clone()
            dropout_mask[torch.randperm(len(dropout_mask))[:n_nonzero_features]] = 1
            dropout_mask = dropout_mask.reshape(mask_size)
            if _CONSTANTS.MODALITY2_LOSS == "mse":
                return torch.sum((real - pred).pow(2) * dropout_mask) / torch.sum(
                    dropout_mask
                )
            elif _CONSTANTS.MODALITY2_LOSS == "bce":
                class_weight = (
                    dropout_mask.shape[0] * dropout_mask.shape[1] - n_nonzero_features
                ) / n_nonzero_features
                return torch.nn.BCEWithLogitsLoss(
                    reduce="mean", pos_weight=class_weight
                )(pred, real)
        else:
            if _CONSTANTS.MODALITY2_LOSS == "mse":
                return F.mse_loss(pred, real, reduction="mean")
            elif _CONSTANTS.MODALITY2_LOSS == "bce":
                return torch.nn.BCEWithLogitsLoss(reduce="mean")(pred, real)


def plot_emb_mod(pred, mod, batch):
    pred = np.concatenate(pred)
    mod = np.concatenate(mod).flatten()
    batch = np.concatenate(batch).flatten()

    ad = sc.AnnData(pred, obs=pd.DataFrame({"mod": mod, "batch": batch}))
    ad.obs["mod"] = ad.obs["mod"].astype("category")
    ad.obs["batch"] = ad.obs["batch"].astype("category")
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)
    sc.pl.umap(ad, color=["mod", "batch"])


def plot_emb_batch(pred, batch):
    pred = np.concatenate(pred)
    batch = np.concatenate(batch).flatten()
    ad = sc.AnnData(pred, obs=pd.DataFrame({"batch": batch}))
    ad.obs["batch"] = ad.obs["batch"].astype("category")
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)
    sc.pl.umap(ad, color=["batch"])


@dataclass
class MVAEParams:
    """
    beta - KL divergence term hyperparam for MVAE
    """

    n_layers = 2
    n_hidden = 800
    z_dim = 200
    beta = 0.1
    dropout = 0.1
    z_dropout = 0.3
    encode_covariates = False
    batch_size = 32


class FullyConnectedLayers(nn.Sequential):
    """
    Architecture:
    """

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
            layers.append(nn.PReLU())
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

        self.n_batch_mod1 = mdata.mod[_CONSTANTS.MODALITY1_KEY].uns["_scvi"][
            "summary_stats"
        ]["n_batch"]

        self.n_batch_mod2 = mdata.mod[_CONSTANTS.MODALITY2_KEY].uns["_scvi"][
            "summary_stats"
        ]["n_batch"]

        print("N batches for mod1: ", str(self.n_batch_mod1))
        print("N batches for mod2: ", str(self.n_batch_mod2))

        rna_shape = mdata.mod[_CONSTANTS.MODALITY1_KEY].shape
        msi_shape = mdata.mod[_CONSTANTS.MODALITY2_KEY].shape

        print(rna_shape)
        print(msi_shape)

        self.rna = ModalityLayers(rna_shape[1], self.n_batch_mod1, params)
        self.msi = ModalityLayers(msi_shape[1], self.n_batch_mod2, params)
        self.poe = PoE()

    def encode(
        self,
        modality_layers: ModalityLayers,
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
        y = modality_layers.encoder(X, batch_id, *categorical_input)

        return modality_layers.sample_latent(y, batch_id)

    def decode(self, z, batch_id1, batch_id2, modality=None, cat_covs=None):
        """
        Decode data from latent space.

        Parameters
        ----------
        z
            data embedded in latent space
        batch_index
            batch information for samples
        cat_covs
            categorical covariates.
        Returns
        -------
        X_rec
            decoded data
        """
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if modality == _CONSTANTS.MODALITY1_KEY:
            X = self.decoder1(z, batch_id1, *categorical_input)
            X = self.final1(X)
        elif modality == _CONSTANTS.MODALITY2_KEY:
            X = self.decoder2(z, batch_id2, *categorical_input)
            X = self.final2(X)

        return X

    @torch.no_grad()
    def to_latent(
        self,
        adata,
        mod1_obsm=None,
        mod2_obsm=None,
        batch_size=1024,
        modality=None,
        indices=None,
        return_mean=False,
        use_gpu=True,
    ):
        """
        Project data into latent space. Inspired by SCVI.

        Parameters
        ----------
        adata
            scanpy single-cell dataset
        indices
            indices of the subset of cells to be encoded
        return_mean
            whether to use the mean of the multivariate gaussian or samples
        """
        dev = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.to(dev)
        #        sc_dl, _ = _anndata_loader(adata, mod1_obsm=mod1_obsm, mod2_obsm=mod2_obsm, batch_size=batch_size, shuffle=False)
        # train_mdata, test_mdata = _anndata_splitter(adata, train_size=1)
        train_loader, train_loader_pairs = _anndata_loader(
            adata,
            mod1_obsm=mod1_obsm,
            mod2_obsm=mod2_obsm,
            batch_size=self.batch_size,
            shuffle=False,
        )
        #        test_loader, test_loader_pairs = _anndata_loader(test_mdata, mod1_obsm=mod1_obsm, mod2_obsm=mod2_obsm, batch_size=self.batch_size, shuffle=False)

        latent_z = []
        latent_z1_s = []
        latent_z2_s = []
        latent_z1_p = []
        latent_z2_p = []
        latent_z1_p_mod = []
        latent_z2_p_mod = []
        with torch.no_grad():
            self.eval()
            for tensors in tqdm(train_loader):
                tensors = {k: v.to(dev) for k, v in tensors.items()}
                model_output = self.forward(tensors)

                latent_z += [model_output["z"].cpu()]
                latent_z1_s += [model_output["z1_s"].cpu().numpy()]
                latent_z2_s += [model_output["z2_s"].cpu().numpy()]
                latent_z1_p += [model_output["z1_p"].cpu().numpy()]
                latent_z2_p += [model_output["z2_p"].cpu().numpy()]
                latent_z1_p_mod += [model_output["z1_p_mod"].cpu().numpy()]
                latent_z2_p_mod += [model_output["z2_p_mod"].cpu().numpy()]

        return (
            latent_z,
            latent_z1_s,
            latent_z2_s,
            latent_z1_p,
            latent_z1_p_mod,
            latent_z2_p,
            latent_z2_p_mod,
        )  # , train_mdata

    def predict(
        self,
        adata,
        mod1_obsm=None,
        mod2_obsm=None,
        batch_size=1024,
        modality=None,
        indices=None,
        return_mean=False,
        use_gpu=True,
    ):
        """
        Project data into latent space. Inspired by SCVI.

        Parameters
        ----------
        adata
            scanpy single-cell dataset
        indices
            indices of the subset of cells to be encoded
        return_mean
            whether to use the mean of the multivariate gaussian or samples
        """
        dev = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.to(dev)
        #        train_mdata, test_mdata = _anndata_splitter(adata, train_size=1)
        train_loader, train_loader_pairs = _anndata_loader(
            adata,
            mod1_obsm=mod1_obsm,
            mod2_obsm=mod2_obsm,
            batch_size=self.batch_size,
            shuffle=False,
        )

        x1_poe = []
        x2_poe = []
        x1 = []
        x2 = []
        x1_2 = []
        x2_1 = []
        x1_batch_free = []
        x2_batch_free = []
        with torch.no_grad():
            self.eval()
            for tensors in tqdm(train_loader):
                tensors = {k: v.to(dev) for k, v in tensors.items()}
                model_output = self.forward(tensors)
                x1_poe += [model_output["x1_poe"].detach().cpu()]
                x2_poe += [model_output["x2_poe"].detach().cpu()]
                x1 += [model_output["x1"].detach().cpu()]
                x2 += [model_output["x2"].detach().cpu()]
                x1_2 += [model_output["x1_2"].detach().cpu()]
                x2_1 += [model_output["x2_1"].detach().cpu()]
                x1_batch_free += [model_output["x1_batch_free"].detach().cpu()]
                x2_batch_free += [model_output["x2_batch_free"].detach().cpu()]

        return x1_poe, x2_poe, x1, x2, x1_2, x2_1, x1_batch_free, x2_batch_free

    def forward(self, tensors: MultimodalDatasetItemT):
        input = MultimodalDatasetItem(**tensors)
        mod_id = input.mod_id

        mu, logvar = initialize_latent([1, mod_id.shape[0], self.z_dim], use_cuda=True)

        rna_latent_p = initialize_latent((mod_id.shape[0], self.z_dim), use_cuda=True)
        rna_latent_mod = initialize_latent((mod_id.shape[0], self.z_dim), use_cuda=True)
        rna_latent_s = initialize_latent((mod_id.shape[0], self.z_dim), use_cuda=True)
        msi_latent_p = initialize_latent((mod_id.shape[0], self.z_dim), use_cuda=True)
        msi_latent_mod = initialize_latent((mod_id.shape[0], self.z_dim), use_cuda=True)
        msi_latent_s = initialize_latent((mod_id.shape[0], self.z_dim), use_cuda=True)

        mod_idxs_1 = (mod_id == 1) | (mod_id == 3)
        mod_idxs_2 = (mod_id == 2) | (mod_id == 3)

        rna_post_p, rna_post_mod, rna_post_s = self.encode(
            self.rna,
            input.rna,
            mod_idxs_1,
            input.batch_id1,
            input.extra_categorical_covs,
        )
        msi_post_p, msi_post_mod, msi_post_s = self.encode(
            self.msi,
            input.msi,
            mod_idxs_2,
            input.batch_id2,
            input.extra_categorical_covs,
        )
        for prior, post in zip(
            [rna_latent_p, rna_latent_mod, rna_latent_s],
            [rna_post_p, rna_post_mod, rna_post_s],
        ):
            prior.update(post)

        for prior, post in zip(
            [msi_latent_p, msi_latent_mod, msi_latent_s],
            [msi_post_p, msi_post_mod, msi_post_s],
        ):
            prior.update(post)

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

        input_decode1_poe = self._get_generative_input(
            tensors, z1_p + z_poe
        )  # z1_p + z_poe + z1_p_mod
        input_decode2_poe = self._get_generative_input(
            tensors, z2_p + z_poe
        )  # z2_p + z_poe + z2_p_mod

        input_decode1 = self._get_generative_input(
            tensors, z1_p + z1_p_mod
        )  # z1_p + z1_p_mod + z1_s
        input_decode2 = self._get_generative_input(
            tensors, z2_p + z2_p_mod
        )  # z2_p + z2_p_mod + z2_s

        input_decode2_1 = self._get_generative_input(
            tensors, z1_p + z2_s
        )  # z1_p + z2_s + z1_p_mod
        input_decode1_2 = self._get_generative_input(
            tensors, z2_p + z1_s
        )  # z2_p + z1_s + z2_p_mod

        input_decode_batch_free_1 = self._get_generative_input(
            tensors, z_poe + z1_p_mod
        )
        input_decode_batch_free_2 = self._get_generative_input(
            tensors, z_poe + z2_p_mod
        )

        #########################################################
        input_decode_batch_only1 = self._get_generative_input(tensors, z1_p)
        input_decode_batch_only2 = self._get_generative_input(tensors, z2_p)

        X1_batch_only = self.decode(
            **input_decode_batch_only1, modality=_CONSTANTS.MODALITY1_KEY
        )
        X2_batch_only = self.decode(
            **input_decode_batch_only2, modality=_CONSTANTS.MODALITY2_KEY
        )
        #########################################################

        X1_poe = self.decode(**input_decode1_poe, modality=_CONSTANTS.MODALITY1_KEY)
        X2_poe = self.decode(**input_decode2_poe, modality=_CONSTANTS.MODALITY2_KEY)

        X1 = self.decode(**input_decode1, modality=_CONSTANTS.MODALITY1_KEY)
        X2 = self.decode(**input_decode2, modality=_CONSTANTS.MODALITY2_KEY)

        # Translation losses
        X1_2 = self.decode(**input_decode1_2, modality=_CONSTANTS.MODALITY2_KEY)
        X2_1 = self.decode(**input_decode2_1, modality=_CONSTANTS.MODALITY1_KEY)

        X1_batch_free = self.decode(
            **input_decode_batch_free_1, modality=_CONSTANTS.MODALITY1_KEY
        )
        X2_batch_free = self.decode(
            **input_decode_batch_free_2, modality=_CONSTANTS.MODALITY2_KEY
        )

        return dict(
            x1_poe=X1_poe,
            x2_poe=X2_poe,
            x1=X1,
            x2=X2,
            x1_2=X1_2,
            x2_1=X2_1,
            x1_batch_free=X1_batch_free,
            x2_batch_free=X2_batch_free,
            mu=mu,
            mu1_p=mu1_p,
            mu1_p_mod=mu1_p_mod,
            mu1_s=mu1_s,
            mu2_p=mu2_p,
            mu2_p_mod=mu2_p_mod,
            mu2_s=mu2_s,
            logvar=logvar,
            logvar1_p=logvar1_p,
            logvar1_p_mod=logvar1_p_mod,
            logvar1_s=logvar1_s,
            logvar2_p=logvar2_p,
            logvar2_p_mod=logvar2_p_mod,
            logvar2_s=logvar2_s,
            z=z_poe,
            z1_p=z1_p,
            z1_s=z1_s,
            z2_p=z2_p,
            z2_s=z2_s,
            z1_p_mod=z1_p_mod,
            z2_p_mod=z2_p_mod,
            x1_batch_only=X1_batch_only,
            x2_batch_only=X2_batch_only,
        )

    # , batch_pred_mod1=batch_pred_mod1, batch_pred_mod2=batch_pred_mod2)

    def reconstruct_masking_latent_dim(
        self,
        adata,
        to_mask="z_shared",
        batch_size=1024,
        mod1_obsm=None,
        mod2_obsm=None,
        modality=None,
        indices=None,
        return_mean=False,
        use_gpu=True,
    ):
        """
        Equivalent to the predict, but modifying the forward passso that the specied latent dim is set to 0s
        Valid options are:
            'z_shared', 'z1_p', 'z1_p_mod', 'z2_p', 'z2_p_mod'
        """

        train_loader, train_loader_pairs = _anndata_loader(
            adata,
            mod1_obsm=mod1_obsm,
            mod2_obsm=mod2_obsm,
            batch_size=self.batch_size,
            shuffle=False,
        )
        dev = torch.device("cuda") if self.use_cuda else torch.device("cpu")

        x1_poe = []
        x2_poe = []
        x1 = []
        x2 = []
        x1_2 = []
        x2_1 = []
        x1_batch_free = []
        x2_batch_free = []
        with torch.no_grad():
            self.eval()
            for tensors in tqdm(train_loader):
                tensors = {k: v.to(dev) for k, v in tensors.items()}

                mod_id = tensors["mod_id"]
                input_encode = self._get_inference_input(tensors)

                mu, logvar = prior_expert(
                    (1, mod_id.shape[0], self.z_dim), use_cuda=True
                )

                # Prior for private representation
                mu1_p, logvar1_p = prior_expert(
                    (mod_id.shape[0], self.z_dim), use_cuda=True
                )
                mu1_p_mod, logvar1_p_mod = prior_expert(
                    (mod_id.shape[0], self.z_dim), use_cuda=True
                )
                mu2_p, logvar2_p = prior_expert(
                    (mod_id.shape[0], self.z_dim), use_cuda=True
                )
                mu2_p_mod, logvar2_p_mod = prior_expert(
                    (mod_id.shape[0], self.z_dim), use_cuda=True
                )
                z1_p = self.sample_latent(mu1_p, logvar1_p)
                z1_p_mod = self.sample_latent(mu1_p_mod, logvar1_p_mod)
                z2_p = self.sample_latent(mu2_p, logvar2_p)
                z2_p_mod = self.sample_latent(mu2_p_mod, logvar2_p_mod)

                # Prior for shared representation
                mu1_s, logvar1_s = prior_expert(
                    (mod_id.shape[0], self.z_dim), use_cuda=True
                )
                mu2_s, logvar2_s = prior_expert(
                    (mod_id.shape[0], self.z_dim), use_cuda=True
                )
                z1_s = self.sample_latent(mu1_s, logvar1_s)
                z2_s = self.sample_latent(mu2_s, logvar2_s)

                mod_idxs_1 = (mod_id == 1) | (mod_id == 3)
                mod_idxs_2 = (mod_id == 2) | (mod_id == 3)

                #   self.encode returns: z_p, mu_p, logvar_p, z_p_mod, mu_p_mod, logvar_p_mod, z_s, mu_s, logvar_s

                (
                    z1_p[mod_idxs_1, :],
                    mu1_p[mod_idxs_1, :],
                    logvar1_p[mod_idxs_1, :],
                    z1_p_mod[mod_idxs_1, :],
                    mu1_p_mod[mod_idxs_1, :],
                    logvar1_p_mod[mod_idxs_1, :],
                    z1_s[mod_idxs_1, :],
                    mu1_s[mod_idxs_1, :],
                    logvar1_s[mod_idxs_1, :],
                ) = self.encode(
                    modality=_CONSTANTS.MODALITY1_KEY,
                    mod_idxs_1=mod_idxs_1,
                    mod_idxs_2=mod_idxs_2,
                    **input_encode,
                )
                (
                    z2_p[mod_idxs_2, :],
                    mu2_p[mod_idxs_2, :],
                    logvar2_p[mod_idxs_2, :],
                    z2_p_mod[mod_idxs_2, :],
                    mu2_p_mod[mod_idxs_2, :],
                    logvar2_p_mod[mod_idxs_2, :],
                    z2_s[mod_idxs_2, :],
                    mu2_s[mod_idxs_2, :],
                    logvar2_s[mod_idxs_2, :],
                ) = self.encode(
                    modality=_CONSTANTS.MODALITY2_KEY,
                    mod_idxs_1=mod_idxs_1,
                    mod_idxs_2=mod_idxs_2,
                    **input_encode,
                )

                mu = torch.cat((mu, mu1_s.unsqueeze(0), mu2_s.unsqueeze(0)), dim=0)
                logvar = torch.cat(
                    (logvar, logvar1_s.unsqueeze(0), logvar2_s.unsqueeze(0)), dim=0
                )

                mu, logvar = self.poe(mu, logvar)
                z_poe = self.sample_latent(mu, logvar)

                if to_mask == "z_shared":
                    z_poe = torch.zeros_like(z_poe)
                    z1_s = torch.zeros_like(z1_s)
                    z2_s = torch.zeros_like(z2_s)

                elif to_mask == "z1_p":
                    z1_p = torch.zeros_like(z1_p)

                elif to_mask == "z1_p_mod":
                    z1_p_mod = torch.zeros_like(z1_p_mod)

                elif to_mask == "z2_p":
                    z2_p = torch.zeros_like(z2_p)

                elif to_mask == "z2_p_mod":
                    z2_p_mod = torch.zeros_like(z2_p_mod)
                #                 else:
                #                     print("Masking nothing")

                input_decode1_poe = self._get_generative_input(
                    tensors, z1_p + z_poe
                )  # z1_p + z_poe + z1_p_mod
                input_decode2_poe = self._get_generative_input(
                    tensors, z2_p + z_poe
                )  # z2_p + z_poe + z2_p_mod

                input_decode1 = self._get_generative_input(
                    tensors, z1_p + z1_p_mod
                )  # z1_p + z1_p_mod + z1_s
                input_decode2 = self._get_generative_input(
                    tensors, z2_p + z2_p_mod
                )  # z2_p + z2_p_mod + z2_s

                input_decode2_1 = self._get_generative_input(
                    tensors, z1_p + z2_s
                )  # z1_p + z2_s + z1_p_mod
                input_decode1_2 = self._get_generative_input(
                    tensors, z2_p + z1_s
                )  # z2_p + z1_s + z2_p_mod

                input_decode_batch_free_1 = self._get_generative_input(
                    tensors, z_poe + z1_p_mod
                )
                input_decode_batch_free_2 = self._get_generative_input(
                    tensors, z_poe + z2_p_mod
                )

                X1_poe = self.decode(
                    **input_decode1_poe, modality=_CONSTANTS.MODALITY1_KEY
                )
                X2_poe = self.decode(
                    **input_decode2_poe, modality=_CONSTANTS.MODALITY2_KEY
                )

                X1 = self.decode(**input_decode1, modality=_CONSTANTS.MODALITY1_KEY)
                X2 = self.decode(**input_decode2, modality=_CONSTANTS.MODALITY2_KEY)

                # Translation losses
                X1_2 = self.decode(**input_decode1_2, modality=_CONSTANTS.MODALITY2_KEY)
                X2_1 = self.decode(**input_decode2_1, modality=_CONSTANTS.MODALITY1_KEY)

                X1_batch_free = self.decode(
                    **input_decode_batch_free_1, modality=_CONSTANTS.MODALITY1_KEY
                )
                X2_batch_free = self.decode(
                    **input_decode_batch_free_2, modality=_CONSTANTS.MODALITY2_KEY
                )

                x1_poe.append(X1_poe.detach().cpu())
                x2_poe.append(X2_poe.detach().cpu())
                x1.append(X1.detach().cpu())
                x2.append(X2.detach().cpu())
                x1_2.append(X1_2.detach().cpu())
                x2_1.append(X2_1.detach().cpu())
                x1_batch_free.append(X1_batch_free.detach().cpu())
                x2_batch_free.append(X2_batch_free.detach().cpu())

        return x1_poe, x2_poe, x1, x2, x1_2, x2_1, x1_batch_free, x2_batch_free

    def shared_loss(self, model_input, model_output):
        """Initializes Maximum Mean Discrepancy(MMD) between source_features and target_features.
        - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.
        Parameters
        ----------
        source_features: torch.Tensor
             Tensor with shape [batch_size, z_dim]
        target_features: torch.Tensor
             Tensor with shape [batch_size, z_dim]
        Returns
        -------
        Returns the computed MMD between x and y.
        """
        (
            mu,
            mu1_p,
            mu2_p,
            logvar,
            logvar1_p,
            logvar2_p,
            mu1_p_mod,
            mu2_p_mod,
            logvar1_p_mod,
            logvar2_p_mod,
        ) = (
            model_output["mu"],
            model_output["mu1_p"],
            model_output["mu2_p"],
            model_output["logvar"],
            model_output["logvar1_p"],
            model_output["logvar2_p"],
            model_output["mu1_p_mod"],
            model_output["mu2_p_mod"],
            model_output["logvar1_p_mod"],
            model_output["logvar2_p_mod"],
        )
        mod1_features, mod2_features = model_output["z1_s"], model_output["z2_s"]
        y_pred1_poe, y_pred1_2, y_true1 = (
            model_output["x1_poe"],
            model_output["x1_2"],
            torch.squeeze(model_input[_CONSTANTS.MODALITY1_KEY]),
        )
        y_pred2_poe, y_pred2_1, y_true2 = (
            model_output["x2_poe"],
            model_output["x2_1"],
            torch.squeeze(model_input[_CONSTANTS.MODALITY2_KEY]),
        )
        mod = model_input["mod_id"]

        mse_mod1_2 = loss_mod(
            y_pred1_2, y_true2, mod, modality=_CONSTANTS.MODALITY2_KEY, dropout=True
        )
        mse_mod2_1 = loss_mod(
            y_pred2_1, y_true1, mod, modality=_CONSTANTS.MODALITY1_KEY, dropout=True
        )

        mse_mod1_poe = loss_mod(
            y_pred1_poe, y_true1, mod, modality=_CONSTANTS.MODALITY1_KEY, dropout=True
        )
        mse_mod2_poe = loss_mod(
            y_pred2_poe, y_true2, mod, modality=_CONSTANTS.MODALITY2_KEY, dropout=True
        )

        # kld1_p = -0.5 * torch.mean(1. + logvar1_p - mu1_p.pow(2) - logvar1_p.exp(), )
        # kld2_p = -0.5 * torch.mean(1. + logvar2_p - mu2_p.pow(2) - logvar2_p.exp(), )
        kld = -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp(),
        )

        alphas = [
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            100,
            1e3,
            1e4,
            1e5,
            1e6,
        ]
        if self.use_cuda:
            dev = "cuda"
        else:
            dev = "cpu"
        alphas = Variable(torch.FloatTensor(alphas)).to(device=dev)

        mmd = torch.mean(gaussian_kernel_matrix(mod1_features, mod1_features, alphas))
        mmd += torch.mean(gaussian_kernel_matrix(mod2_features, mod2_features, alphas))
        mmd -= 2 * torch.mean(
            gaussian_kernel_matrix(mod2_features, mod1_features, alphas)
        )

        cos = nn.CosineSimilarity()
        cos_loss1 = torch.mean(cos(model_output["z1_p"], model_output["z1_s"]).abs())
        cos_loss2 = torch.mean(cos(model_output["z2_p"], model_output["z2_s"]).abs())

        return (
            torch.mean(mse_mod1_2)
            + torch.mean(mse_mod2_1)
            + torch.mean(mse_mod1_poe)
            + torch.mean(mse_mod2_poe)
            + self.beta * torch.mean(kld),
            mmd,
            torch.mean(mse_mod1_2),
            torch.mean(mse_mod2_1),
            torch.mean(mse_mod1_poe),
            torch.mean(mse_mod2_poe),
            self.beta * torch.mean(kld),
            cos_loss1,
            cos_loss2,
        )

    def private_loss(self, model_input, model_output):
        """
        Custom loss for beta-VAE
        Parameters
        ----------
        model_input
            dict with input values
        model_output
            dict with output values
        Returns
        -------
        loss value for current batch
        """
        # Parse values
        (
            mu1_p,
            mu1_s,
            mu2_p,
            mu2_s,
            logvar1_p,
            logvar1_s,
            logvar2_p,
            logvar2_s,
            mu1_p_mod,
            mu2_p_mod,
            logvar1_p_mod,
            logvar2_p_mod,
        ) = (
            model_output["mu1_p"],
            model_output["mu1_s"],
            model_output["mu2_p"],
            model_output["mu2_s"],
            model_output["logvar1_p"],
            model_output["logvar1_s"],
            model_output["logvar2_p"],
            model_output["logvar2_s"],
            model_output["mu1_p_mod"],
            model_output["mu2_p_mod"],
            model_output["logvar1_p_mod"],
            model_output["logvar2_p_mod"],
        )
        y_pred1, y_true1 = model_output["x1"], torch.squeeze(
            model_input[_CONSTANTS.MODALITY1_KEY]
        )
        y_pred2, y_true2 = model_output["x2"], torch.squeeze(
            model_input[_CONSTANTS.MODALITY2_KEY]
        )
        mod = model_input["mod_id"]
        # Get Loss

        kld1_p = -0.5 * torch.mean(
            1.0 + logvar1_p - mu1_p.pow(2) - logvar1_p.exp(),
        )  # * 0.1
        kld1_p_mod = -0.5 * torch.mean(
            1.0 + logvar1_p_mod - mu1_p_mod.pow(2) - logvar1_p_mod.exp(),
        )
        kld1_s = -0.5 * torch.mean(
            1.0 + logvar1_s - mu1_s.pow(2) - logvar1_s.exp(),
        )

        kld2_p = -0.5 * torch.mean(
            1.0 + logvar2_p - mu2_p.pow(2) - logvar2_p.exp(),
        )  # * 0.1
        kld2_p_mod = -0.5 * torch.mean(
            1.0 + logvar2_p_mod - mu2_p_mod.pow(2) - logvar2_p_mod.exp(),
        )
        kld2_s = -0.5 * torch.mean(
            1.0 + logvar2_s - mu2_s.pow(2) - logvar2_s.exp(),
        )

        mse_mod1 = loss_mod(
            y_pred1, y_true1, mod, modality=_CONSTANTS.MODALITY1_KEY, dropout=True
        )
        mse_mod2 = loss_mod(
            y_pred2, y_true2, mod, modality=_CONSTANTS.MODALITY2_KEY, dropout=True
        )

        ################################################################################################
        y_pred_batch1 = model_output["x1_batch_only"]
        y_pred_batch2 = model_output["x2_batch_only"]

        mse_mod_batch1 = loss_mod(
            y_pred_batch1, y_true1, mod, modality=_CONSTANTS.MODALITY1_KEY, dropout=True
        )
        mse_mod_batch2 = loss_mod(
            y_pred_batch2, y_true2, mod, modality=_CONSTANTS.MODALITY2_KEY, dropout=True
        )

        # NOW THEY ARE NOT BEING USED IN THE LOSS
        ################################################################################################

        return (
            torch.mean(mse_mod1)
            + torch.mean(mse_mod2)
            + self.beta
            * (
                torch.mean(kld1_p)
                + torch.mean(kld1_s)
                + torch.mean(kld2_p)
                + torch.mean(kld2_s)
            ),
            torch.mean(mse_mod1),
            torch.mean(mse_mod2),
            torch.mean(self.beta * kld1_p),
            torch.mean(self.beta * kld1_s),
            torch.mean(self.beta * kld2_p),
            torch.mean(self.beta * kld2_s),
            torch.mean(self.beta * kld1_p_mod),
            torch.mean(self.beta * kld2_p_mod),
            torch.mean(mse_mod_batch1),
            torch.mean(mse_mod_batch2),
        )

    def train_mvae(
        self,
        mdata,
        params=TrainParams(),
        **kwargs,
    ):
        train_mdata, test_mdata = split_into_train_test(
            mdata,
            params.train_size,
            sample=params.leave_sample_out,
            batch_split=params.batch_split,
        )

        train_loader, train_loader_pairs = mudata_to_dataloader(
            train_mdata,
            batch_size=params.batch_size,
            shuffle=params.shuffle,
        )
        test_loader, test_loader_pairs = mudata_to_dataloader(
            test_mdata,
            batch_size=params.batch_size,
            shuffle=params.shuffle,
        )

        self.to(self.device)

        self.epoch_history = self._train_model(
            train_loader,
            train_loader_pairs,
            test_loader,
            test_loader_pairs,
            params,
        )
        self.eval()
        return None

    def _train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        train_loader_pairs: torch.utils.data.DataLoader,
        test_loader=None,
        test_loader_pairs=None,
        params: TrainParams = TrainParams(),
    ):
        # Initialize Tensorboard summary writer
        writer = SummaryWriter(
            "logs/mvae" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        epoch_hist = {"train_loss": [], "valid_loss": []}
        optimizer = optim.Adam(
            self.parameters(), lr=params.learning_rate, weight_decay=5e-4
        )
        train_ES = EarlyStopping(
            patience=params.train_patience, verbose=True, mode="train"
        )
        if test_loader:
            valid_ES = EarlyStopping(
                patience=params.test_patience, verbose=True, mode="valid"
            )

        # Train
        it = 0
        for epoch in range(params.n_epochs):
            torch.set_num_threads(16)
            self.train()
            for model_input, model_input_pairs in tqdm(
                zip(train_loader, train_loader_pairs), total=len(train_loader)
            ):
                optimizer.zero_grad()

                loss = Loss()

                # Send input to device
                model_input = {k: v.to(self.device) for k, v in model_input.items()}
                model_input_pairs = {
                    k: v.to(self.device) for k, v in model_input_pairs.items()
                }
                optimizer.zero_grad()

                model_output = self.forward(model_input)
                model_output_pairs = self.forward(model_input_pairs)

                (
                    loss_private,
                    loss_rec_rna,
                    loss_rec_msi,
                    loss_kl1_p,
                    loss_kl1_s,
                    loss_kl2_p,
                    loss_kl2_s,
                    loss_kl1_p_mod,
                    loss_kl2_p_mod,
                    loss_batch1,
                    loss_batch2,
                ) = self.private_loss(model_input, model_output)
                (
                    loss_shared,
                    loss_mmd,
                    loss_trls_msi,
                    loss_trls_rna,
                    loss_rec_rna_poe,
                    loss_rec_msi_poe,
                    loss_kl,
                    loss_cos1,
                    loss_cos2,
                ) = self.shared_loss(model_input_pairs, model_output_pairs)

                loss = loss_private + loss_shared  # + loss_batch_mod1 + loss_batch_mod2
                loss_value += loss.item()
                loss_rec_rna_v += loss_rec_rna.item()
                loss_rec_msi_v += loss_rec_msi.item()
                loss_trls_msi_v += loss_trls_msi.item()
                loss_trls_rna_v += loss_trls_rna.item()
                loss_rec_rna_poe_v += loss_rec_rna_poe.item()
                loss_rec_msi_poe_v += loss_rec_msi_poe.item()
                loss_kl_v += loss_kl.item()
                loss_kl1_p_v += loss_kl1_p.item()
                loss_kl1_p_mod_v += loss_kl1_p_mod.item()
                loss_kl1_s_v += loss_kl1_s.item()
                loss_kl2_p_v += loss_kl2_p.item()
                loss_kl2_p_mod_v += loss_kl2_p_mod.item()
                loss_kl2_s_v += loss_kl2_s.item()
                loss_shared_v += loss_shared.item()
                loss_private_v += loss_private.item()
                loss_mmd_v += loss_mmd.item()
                loss_cos1_v += loss_cos1
                loss_cos2_v += loss_cos2

                ####################################
                loss_batch_1_v += loss_batch1.item()
                loss_batch_2_v += loss_batch2.item()
                ####################################

                loss.backward()
                optimizer.step()

                writer.add_scalar("PoE_training/Loss", loss_value, it)
                writer.add_scalar("PoE_training/mse_rna", loss_rec_rna_v, it)
                writer.add_scalar("PoE_training/mse_msi", loss_rec_msi_v, it)
                writer.add_scalar("PoE_training/mse_rna_poe", loss_rec_rna_poe_v, it)
                writer.add_scalar("PoE_training/mse_msi_poe", loss_rec_msi_poe_v, it)
                writer.add_scalar("PoE_training/kl_loss", loss_kl_v, it)
                writer.add_scalar("PoE_training/kl1_p_loss", loss_kl1_p_v, it)
                writer.add_scalar("PoE_training/kl1_s_loss", loss_kl1_s_v, it)
                writer.add_scalar("PoE_training/kl1_p_mod_loss", loss_kl1_p_mod_v, it)
                writer.add_scalar("PoE_training/kl2_p_loss", loss_kl2_p_v, it)
                writer.add_scalar("PoE_training/kl2_s_loss", loss_kl2_s_v, it)
                writer.add_scalar("PoE_training/kl2_p_mod_loss", loss_kl2_p_mod_v, it)
                writer.add_scalar("PoE_training/mmd_loss", loss_mmd_v, it)
                writer.add_scalar("PoE_training/shared_loss", loss_shared_v, it)
                writer.add_scalar("PoE_training/private_loss", loss_private_v, it)
                writer.add_scalar("PoE_training/trls_rna_loss", loss_trls_rna_v, it)
                writer.add_scalar("PoE_training/trls_msi_loss", loss_trls_msi_v, it)
                writer.add_scalar("PoE_training/cos1_loss", loss_cos1_v, it)
                writer.add_scalar("PoE_training/cos2_loss", loss_cos2_v, it)
                ##############################################################################
                writer.add_scalar("PoE_training/loss_batch1", loss_batch_1_v, it)
                writer.add_scalar("PoE_training/loss_batch2", loss_batch_2_v, it)
                ##############################################################################

                it += 1

            # Get epoch loss
            epoch_loss = loss_value / len(train_loader.dataset.indices)
            epoch_hist["train_loss"].append(epoch_loss)
            train_ES(epoch_loss)
            # Eval
            if test_loader:
                self.eval()
                torch.save(self.state_dict(), "mvae_params.pt")
                test_dict = self._test_model(test_loader, test_loader_pairs, device)
                test_loss = test_dict["loss"]
                epoch_hist["valid_loss"].append(test_loss)
                valid_ES(test_loss)
                writer.add_scalar("PoE_training/test_loss", test_loss, epoch + 1)
                writer.add_scalar(
                    "PoE_training/test_loss_shared", test_dict["loss_shared"], epoch + 1
                )
                writer.add_scalar(
                    "PoE_training/test_loss_batch_mod1",
                    test_dict["loss_batch_mod1"],
                    epoch + 1,
                )
                writer.add_scalar(
                    "PoE_training/test_loss_batch_mod2",
                    test_dict["loss_batch_mod2"],
                    epoch + 1,
                )

                print(
                    "[Epoch %d] | loss: %.3f | loss_rna: %.3f |loss_msi: %.3f | test_loss: %.3f |"
                    % (
                        epoch + 1,
                        epoch_loss,
                        loss_rec_rna_v / len(train_loader.dataset.indices),
                        loss_rec_msi_v / len(train_loader.dataset.indices),
                        test_loss,
                    ),
                    flush=True,
                )
                if valid_ES.early_stop or train_ES.early_stop:
                    # print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    # break
                    print("", end="")
                else:
                    print(
                        "[Epoch %d] | loss: %.3f |" % (epoch + 1, epoch_loss),
                        flush=True,
                    )
                    # if train_ES.early_stop:
                    # print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    # break
        return epoch_hist

    def _test_model(self, loader, loader_pairs, device):
        """
        Test model on input loader.
        """
        test_dict = {}
        loss_private = 0
        loss_shared = 0
        loss_batch_mod1 = 0
        loss_batch_mod2 = 0
        self.eval()
        i = 0
        latents = []
        latents_1s = []
        latents_2s = []
        latents_1p = []
        latents_1p_mod = []
        latents_2p = []
        latents_2p_mod = []
        mod_ids = []
        batch1_ids = []
        batch2_ids = []
        with torch.no_grad():
            for data, data_pairs in tqdm(zip(loader, loader_pairs), total=len(loader)):
                data = {k: v.to(device) for k, v in data.items()}
                data_pairs = {k: v.to(device) for k, v in data_pairs.items()}
                model_output = self.forward(data)
                latents.append(model_output["z"].cpu().numpy())
                latents_1p.append(model_output["z1_p"].cpu().numpy())
                latents_2p.append(model_output["z2_p"].cpu().numpy())
                latents_1s.append(model_output["z1_s"].cpu().numpy())
                latents_2s.append(model_output["z2_s"].cpu().numpy())
                latents_1p_mod.append(model_output["z1_p_mod"].cpu().numpy())
                latents_2p_mod.append(model_output["z2_p_mod"].cpu().numpy())
                mod_ids.append(data["mod_id"].cpu().numpy())
                batch1_ids.append(data["batch_id1"].cpu().numpy())
                batch2_ids.append(data["batch_id2"].cpu().numpy())

                model_output_pairs = self.forward(data_pairs)
                loss_private += self.private_loss(data, model_output)[0].item()
                loss_shared += self.shared_loss(data_pairs, model_output_pairs)[
                    0
                ].item()
                i += 1

        #         plot_emb_mod(latents, mod_ids, batch1_ids)
        #         plot_emb_batch(latents_1s, batch1_ids)
        #         plot_emb_batch(latents_2s, batch2_ids)
        #         plot_emb_batch(latents_1p, batch1_ids)
        #         plot_emb_batch(latents_2p, batch2_ids)

        test_dict["loss"] = (loss_private + loss_shared) / i
        test_dict["loss_private"] = loss_private / i
        test_dict["loss_shared"] = loss_shared / i
        test_dict["loss_batch_mod1"] = loss_batch_mod1 / i
        test_dict["loss_batch_mod2"] = loss_batch_mod2 / i
        return test_dict

    def fine_tune_batch_layer(
        self,
        mdata,
        mod1_obsm=None,
        mod2_obsm=None,
        learning_rate=1e-4,
        n_epochs=500,
        train_size=1.0,
        batch_size=128,
        batch_split=None,
        shuffle=True,
        use_gpu=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        learning_rate
            learning rate
        n_epochs
            number of epochs to fine-tune model
        train_size
            a number between 0 and 1 to indicate the proportion of training data. Test size is set to 1-train_size
        batch_size
            number of samples per batch
        shuffle
            whether to shuffle samples or not
        use_gpu
            whether to use GPU
        **kwargs
            other keyword arguments of the _train_model() method, like the early stopping patience
        """
        train_patience = kwargs.get("train_patience", 20)
        test_patience = kwargs.get("test_patience", 20)

        train_mdata, test_mdata = _anndata_splitter(
            mdata, train_size=train_size, batch_split=batch_split
        )
        train_loader, train_loader_pairs = _anndata_loader(
            train_mdata,
            mod1_obsm=mod1_obsm,
            mod2_obsm=mod2_obsm,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )
        test_loader, test_loader_pairs = _anndata_loader(
            test_mdata,
            mod1_obsm=mod1_obsm,
            mod2_obsm=mod2_obsm,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

        dev = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.to(dev)
        # train_loader, test_loader = _scvi_loader(self.adata, train_size=train_size, batch_size=batch_size, use_gpu=use_gpu)
        # Call training method
        self.tune_epoch_history = self._fine_tune_model(
            train_loader=train_loader,
            train_loader_pairs=train_loader_pairs,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            train_patience=train_patience,
            test_patience=test_patience,
            test_loader=test_loader,
            test_loader_pairs=test_loader_pairs,
            device=dev,
        )

        for p in self.parameters():
            p.requires_grad = True

        # Set to eval mode
        self.eval()
        return None

    def _fine_tune_model(
        self,
        train_loader,
        train_loader_pairs,
        learning_rate,
        n_epochs,
        train_patience=10,
        test_patience=10,
        test_loader=False,
        test_loader_pairs=False,
        device=torch.device("cpu"),
    ):
        """
        Retraining of batch layer
        Parameters
        ----------
        train_loader
            loader with training data
        learning_rate
            learning rate for training
        n_epochs
            number of maximum epochs to train the model
        train_patience
            early stopping patience for training loss
        test_patience
            early stopping patience for test loss
        test_loader
            if available, loader with test data
        Returns
        -------
            epoch_hist (dict): Training history
        """
        # Initialize Tensorboard summary writer
        writer = SummaryWriter(
            "logs/fine_tune/fine-tune"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        epoch_hist = {}
        epoch_hist["train_loss"] = []
        epoch_hist["valid_loss"] = []
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        train_ES = EarlyStopping(patience=train_patience, verbose=True, mode="train")
        if test_loader:
            valid_ES = EarlyStopping(patience=test_patience, verbose=True, mode="valid")

        # Train
        it = 0
        # self.eval() # !!!!!
        for p in self.parameters():
            p.requires_grad = False

        for p in self.mean1_private.parameters():
            p.requires_grad = True
        for p in self.logvar1_private.parameters():
            p.requires_grad = True
        for p in self.mean2_private.parameters():
            p.requires_grad = True
        for p in self.logvar2_private.parameters():
            p.requires_grad = True

        for epoch in range(n_epochs):
            torch.set_num_threads(16)
            self.train()
            for model_input, model_input_pairs in tqdm(
                zip(train_loader, train_loader_pairs), total=len(train_loader)
            ):
                optimizer.zero_grad()

                loss_value = 0
                loss_rec_rna_v = 0
                loss_rec_msi_v = 0
                loss_trls_msi_v = 0
                loss_trls_rna_v = 0
                loss_rec_rna_poe_v = 0
                loss_rec_msi_poe_v = 0
                loss_kl_v = 0
                loss_kl1_p_v = 0
                loss_kl1_p_mod_v = 0
                loss_kl1_s_v = 0
                loss_kl2_p_v = 0
                loss_kl2_p_mod_v = 0
                loss_kl2_s_v = 0
                loss_mmd_v = 0
                loss_shared_v = 0
                loss_private_v = 0
                loss_private_mod_v = 0
                loss_batch_mod1_v = 0
                loss_batch_mod2_v = 0
                loss_cos1_v = 0
                loss_cos2_v = 0

                ####################################
                loss_batch_1_v = 0
                loss_batch_2_v = 0
                ####################################

                # Send input to device
                model_input = {k: v.to(device) for k, v in model_input.items()}
                model_input_pairs = {
                    k: v.to(device) for k, v in model_input_pairs.items()
                }
                optimizer.zero_grad()

                model_output = self.forward(model_input)
                model_output_pairs = self.forward(model_input_pairs)

                (
                    loss_private,
                    loss_rec_rna,
                    loss_rec_msi,
                    loss_kl1_p,
                    loss_kl1_s,
                    loss_kl2_p,
                    loss_kl2_s,
                    loss_kl1_p_mod,
                    loss_kl2_p_mod,
                    loss_batch1,
                    loss_batch2,
                ) = self.private_loss(model_input, model_output)
                (
                    loss_shared,
                    loss_mmd,
                    loss_trls_msi,
                    loss_trls_rna,
                    loss_rec_rna_poe,
                    loss_rec_msi_poe,
                    loss_kl,
                    loss_cos1,
                    loss_cos2,
                ) = self.shared_loss(model_input_pairs, model_output_pairs)

                loss = loss_private + loss_shared  # + loss_batch_mod1 + loss_batch_mod2
                loss_value += loss.item()
                loss_rec_rna_v += loss_rec_rna.item()
                loss_rec_msi_v += loss_rec_msi.item()
                loss_trls_msi_v += loss_trls_msi.item()
                loss_trls_rna_v += loss_trls_rna.item()
                loss_rec_rna_poe_v += loss_rec_rna_poe.item()
                loss_rec_msi_poe_v += loss_rec_msi_poe.item()
                loss_kl_v += loss_kl.item()
                loss_kl1_p_v += loss_kl1_p.item()
                loss_kl1_p_mod_v += loss_kl1_p_mod.item()
                loss_kl1_s_v += loss_kl1_s.item()
                loss_kl2_p_v += loss_kl2_p.item()
                loss_kl2_p_mod_v += loss_kl2_p_mod.item()
                loss_kl2_s_v += loss_kl2_s.item()
                loss_shared_v += loss_shared.item()
                loss_private_v += loss_private.item()
                loss_mmd_v += loss_mmd.item()
                loss_cos1_v += loss_cos1
                loss_cos2_v += loss_cos2

                ####################################
                loss_batch_1_v += loss_batch1.item()
                loss_batch_2_v += loss_batch2.item()
                ####################################

                loss.backward()
                optimizer.step()

                writer.add_scalar("fine_tuning/Loss", loss_value, it)
                writer.add_scalar("fine_tuning/mse_rna", loss_rec_rna_v, it)
                writer.add_scalar("fine_tuning/mse_msi", loss_rec_msi_v, it)
                writer.add_scalar("fine_tuning/mse_rna_poe", loss_rec_rna_poe_v, it)
                writer.add_scalar("fine_tuning/mse_msi_poe", loss_rec_msi_poe_v, it)
                writer.add_scalar("fine_tuning/kl_loss", loss_kl_v, it)
                writer.add_scalar("fine_tuning/kl1_p_loss", loss_kl1_p_v, it)
                writer.add_scalar("fine_tuning/kl1_s_loss", loss_kl1_s_v, it)
                writer.add_scalar("fine_tuning/kl1_p_mod_loss", loss_kl1_p_mod_v, it)
                writer.add_scalar("fine_tuning/kl2_p_loss", loss_kl2_p_v, it)
                writer.add_scalar("fine_tuning/kl2_s_loss", loss_kl2_s_v, it)
                writer.add_scalar("fine_tuning/kl2_p_mod_loss", loss_kl2_p_mod_v, it)
                writer.add_scalar("fine_tuning/mmd_loss", loss_mmd_v, it)
                writer.add_scalar("fine_tuning/shared_loss", loss_shared_v, it)
                writer.add_scalar("fine_tuning/private_loss", loss_private_v, it)
                writer.add_scalar("fine_tuning/trls_rna_loss", loss_trls_rna_v, it)
                writer.add_scalar("fine_tuning/trls_msi_loss", loss_trls_msi_v, it)
                writer.add_scalar("fine_tuning/cos1_loss", loss_cos1_v, it)
                writer.add_scalar("fine_tuning/cos2_loss", loss_cos2_v, it)
                ##############################################################################
                writer.add_scalar("fine_tuning/loss_batch1", loss_batch_1_v, it)
                writer.add_scalar("fine_tuning/loss_batch2", loss_batch_2_v, it)
                ##############################################################################

                it += 1

            # Get epoch loss
            epoch_loss = loss_value / len(train_loader.dataset.indices)
            epoch_hist["train_loss"].append(epoch_loss)
            train_ES(epoch_loss)
            # Eval
            if test_loader:
                self.eval()
                torch.save(self.state_dict(), "mvae_params_fine_tuned.pt")
                test_dict = self._test_model(test_loader, test_loader_pairs, device)
                test_loss = test_dict["loss"]
                epoch_hist["valid_loss"].append(test_loss)
                valid_ES(test_loss)
                writer.add_scalar("fine_tuning/test_loss", test_loss, epoch + 1)
                writer.add_scalar(
                    "fine_tuning/test_loss_shared", test_dict["loss_shared"], epoch + 1
                )
                writer.add_scalar(
                    "fine_tuning/test_loss_batch_mod1",
                    test_dict["loss_batch_mod1"],
                    epoch + 1,
                )
                writer.add_scalar(
                    "fine_tuning/test_loss_batch_mod2",
                    test_dict["loss_batch_mod2"],
                    epoch + 1,
                )

                print(
                    "[Epoch %d] | loss: %.3f | loss_rna: %.3f |loss_msi: %.3f | test_loss: %.3f |"
                    % (
                        epoch + 1,
                        epoch_loss,
                        loss_rec_rna_v / len(train_loader.dataset.indices),
                        loss_rec_msi_v / len(train_loader.dataset.indices),
                        test_loss,
                    ),
                    flush=True,
                )
                if valid_ES.early_stop or train_ES.early_stop:
                    # print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    # break
                    print("", end="")
                else:
                    print(
                        "[Epoch %d] | loss: %.3f |" % (epoch + 1, epoch_loss),
                        flush=True,
                    )
                    # if train_ES.early_stop:
                    # print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    # break
        return epoch_hist
