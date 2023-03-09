from typing import Callable, Dict

from src.latent import Latent

from src.types import (
    Modality,
    ModalityOutput,
    ModelInputT,
    ModelOutputT,
    ObsModalityMembership,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_loss_fun(loss_fun: str):
    return {
        "mse": mse,
        "bce": bce,
    }[loss_fun]


def mse(pred: torch.Tensor, real: torch.Tensor, dropout=False) -> torch.Tensor:
    # if not dropout:
    return F.mse_loss(pred, real, reduction="mean")

    dropout_mask = create_dropout_mask(real)
    return torch.sum((real - pred).pow(2) * dropout_mask) / torch.sum(dropout_mask)


def bce(pred: torch.Tensor, real: torch.Tensor, dropout=False) -> torch.Tensor:
    if not dropout:
        return torch.nn.BCEWithLogitsLoss(reduce="mean")(pred, real)

    dropout_mask = create_dropout_mask(real)
    n_nonzero_features = dropout_mask.sum().int()
    class_weight = (
        dropout_mask.shape[0] * dropout_mask.shape[1] - n_nonzero_features
    ) / n_nonzero_features
    return torch.nn.BCEWithLogitsLoss(reduce="mean", pos_weight=class_weight)(
        pred, real
    )


def create_dropout_mask(real: torch.Tensor) -> torch.Tensor:
    dropout_mask = (real != 0).float()
    n_nonzero_features = dropout_mask.sum().int()
    mask_size = dropout_mask.size()
    dropout_mask = dropout_mask.reshape(-1)  # .clone()
    dropout_mask[torch.randperm(len(dropout_mask))[:n_nonzero_features]] = 1
    return dropout_mask.reshape(mask_size)


def mmd(x: torch.Tensor, y: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """
    Computes MMD between x and y.
    """
    x_kernel = gaussian_kernel_matrix(x, x, alphas)
    y_kernel = gaussian_kernel_matrix(y, y, alphas)
    xy_kernel = gaussian_kernel_matrix(x, y, alphas)

    loss = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return loss


def gaussian_kernel_matrix(
    x: torch.Tensor, y: torch.Tensor, alphas: torch.Tensor
) -> torch.Tensor:
    """
    Computes multiscale-RBF kernel between x and y.
    """
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1.0 / (2.0 * alphas)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    return torch.transpose(output, 0, 1)


class Loss:
    _loss = None
    private = None
    shared = None
    mmd = None
    rna = None
    msi = None
    rna_batch = None
    msi_batch = None
    rna_kl_p = None
    rna_kl_mod = None
    rna_kl_s = None
    msi_kl_p = None
    msi_kl_mod = None
    msi_kl_s = None

    translation_loss_msi = None
    translation_loss_rna = None

    recovered_rna_poe = None
    recovered_msi_poe = None

    kl = None
    cos_rna = None
    cos_msi = None

    beta: float
    loss_function: Callable = mse
    dropout: bool = False

    def __init__(self, beta, loss_function="mse", dropout=False):
        self.beta = beta
        self.loss_function = get_loss_fun(loss_function)
        self.dropout = dropout

    @property
    def values(self) -> Dict[str, float]:
        return {
            "loss": self._loss.item(),
            "private": self.private.item(),
            "shared": self.shared.item(),
            "mmd": self.mmd.item(),
            "rna": self.rna.item(),
            "msi": self.msi.item(),
            "rna_batch": self.rna_batch.item(),
            "msi_batch": self.msi_batch.item(),
            "rna_kl_p": self.rna_kl_p.item(),
            "rna_kl_mod": self.rna_kl_mod.item(),
            "rna_kl_s": self.rna_kl_s.item(),
            "msi_kl_p": self.msi_kl_p.item(),
            "msi_kl_mod": self.msi_kl_mod.item(),
            "msi_kl_s": self.msi_kl_s.item(),
            "translation_loss_msi": self.translation_loss_msi.item(),
            "translation_loss_rna": self.translation_loss_rna.item(),
            "recovered_rna_poe": self.recovered_rna_poe.item(),
            "recovered_msi_poe": self.recovered_msi_poe.item(),
            "kl": self.kl.item(),
            "cos_rna": self.cos_rna.item(),
            "cos_msi": self.cos_msi.item(),
        }

    def backward(self) -> None:
        self._loss.backward()

    def calculate_private(
        self,
        model_input: ModelInputT,
        model_output: ModelOutputT,
    ) -> None:
        mod = model_input["mod_id"]
        rna_output = ModalityOutput.from_dict(model_output[Modality.rna.name])
        msi_output = ModalityOutput.from_dict(model_output[Modality.msi.name])
        rna_idxs = (mod == ObsModalityMembership.ONLY_MOD1) | (
            mod == ObsModalityMembership.PAIRED
        )
        msi_idxs = (mod == ObsModalityMembership.ONLY_MOD2) | (
            mod == ObsModalityMembership.PAIRED
        )
        (
            self.rna,
            self.rna_batch,
            self.rna_kl_p,
            self.rna_kl_mod,
            self.rna_kl_s,
        ) = self._loss_mod(model_input[Modality.rna.name], rna_output, rna_idxs)
        (
            self.msi,
            self.msi_batch,
            self.msi_kl_p,
            self.msi_kl_mod,
            self.msi_kl_s,
        ) = self._loss_mod(model_input[Modality.msi.name], msi_output, msi_idxs)

        self.private = (
            self.rna
            + self.msi
            + (self.rna_kl_p + self.rna_kl_s + self.msi_kl_p + self.msi_kl_s)
        )

    def _loss_mod(
        self,
        modality_input,
        modality_output: ModalityOutput,
        mod_idxs,
    ):
        """
        Calculates private loss components (MSE and KL) for one modality.
        """
        kld_p = modality_output.latent_p.kld()
        kld_mod = modality_output.latent_mod.kld()
        kld_s = modality_output.latent_s.kld()

        x_pred = modality_output.x[mod_idxs]
        x_real = torch.squeeze(modality_input[mod_idxs])

        x_pred_batch = modality_output.x_batch_only

        loss = self.loss_function(x_pred, x_real, self.dropout)
        loss_batch = self.loss_function(x_pred_batch, x_real, self.dropout)
        return (
            torch.mean(loss),
            torch.mean(loss_batch),
            self.beta * torch.mean(kld_p),
            self.beta * torch.mean(kld_mod),
            self.beta * torch.mean(kld_s),
        )

    def calculate_shared(
        self, model_input: ModelInputT, model_output: ModelOutputT
    ) -> None:
        """
        Initializes Maximum Mean Discrepancy(MMD) between model_input and output.
        - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.
        """
        rna_output = ModalityOutput.from_dict(model_output[Modality.rna.name])
        msi_output = ModalityOutput.from_dict(model_output[Modality.msi.name])
        mod = model_input["mod_id"]
        rna_idxs = (mod == ObsModalityMembership.ONLY_MOD1) | (
            mod == ObsModalityMembership.PAIRED
        )
        msi_idxs = (mod == ObsModalityMembership.ONLY_MOD2) | (
            mod == ObsModalityMembership.PAIRED
        )
        rna_real = torch.squeeze(model_input[Modality.rna.name])
        msi_real = torch.squeeze(model_input[Modality.msi.name])

        self.translation_loss_msi = torch.mean(
            self.loss_function(
                model_output["rna_msi_loss"][msi_idxs],
                msi_real[msi_idxs],
                dropout=self.dropout,
            )
        )
        self.translation_loss_rna = torch.mean(
            self.loss_function(
                model_output["msi_rna_loss"][rna_idxs],
                rna_real[rna_idxs],
                dropout=self.dropout,
            )
        )

        self.recovered_rna_poe = torch.mean(
            self.loss_function(
                model_output["rna_poe"][rna_idxs],
                rna_real[rna_idxs],
                dropout=self.dropout,
            )
        )
        self.recovered_msi_poe = torch.mean(
            self.loss_function(
                model_output["msi_poe"][msi_idxs],
                msi_real[msi_idxs],
                dropout=self.dropout,
            )
        )

        self.kl = torch.mean(Latent(**model_output["poe_latent"]).kld())

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

        alphas = Variable(torch.FloatTensor(alphas)).to(device=torch.device("cuda"))

        self.mmd = mmd(rna_output.latent_s.z, msi_output.latent_s.z, alphas)

        cos = nn.CosineSimilarity()
        self.cos_rna = torch.mean(
            cos(rna_output.latent_p.z, rna_output.latent_s.z).abs()
        )
        self.cos_msi = torch.mean(
            cos(msi_output.latent_p.z, msi_output.latent_s.z).abs()
        )

        self.shared = (
            self.translation_loss_msi
            + self.translation_loss_rna
            + self.recovered_rna_poe
            + self.recovered_msi_poe
            + self.beta * self.kl
        )

        self._loss = self.private + self.shared
