from typing import Callable, Dict

from src.constants import BATCH_KEY

from src.loss import get_loss_fun, mse, compute_lisi, nearest_neighbors

from src.latent import Latent

from src.mvae.types import (
    Modality,
    ModalityOutput,
    ModelInputT,
    ModelOutputT,
)
import torch


class LossCalculator:
    private = None
    shared = None
    batch_integration = None
    batch_integration_rna = None
    batch_integration_msi = None
    batch_loss = None
    rna = None
    msi = None
    rna_poe = None
    msi_poe = None
    rna_msi_loss = None
    msi_rna_loss = None
    rna_kl_p = None
    rna_kl_mod = None
    rna_kl_s = None
    msi_kl_p = None
    msi_kl_mod = None
    msi_kl_s = None

    loss_rna_msi = None
    loss_msi_rna = None

    kl = None

    beta: float
    gamma: float = 1.0
    delta: float = 0.1
    loss_function: Callable = mse
    dropout: bool = False
    batch_num: int = 0

    def __init__(
        self, beta, gamma=1.0, delta=1.0, loss_function="mse", dropout=True, batch_num=0
    ):
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.loss_function = get_loss_fun(loss_function)
        self.dropout = dropout
        self.batch_num = batch_num

    @property
    def total_loss(self) -> torch.Tensor:
        total = self.private + self.shared
        if self.batch_integration:
            total += self.batch_integration
        return total

    @property
    def values(self) -> Dict[str, float]:
        vals = {
            "private": self.private.item(),
            "shared": self.shared.item(),
            "rna_mse": self.rna.item(),
            "msi_mse": self.msi.item(),
            "recovered_rna_poe": self.rna_poe.item(),
            "recovered_msi_poe": self.msi_poe.item(),
            "loss_msi_rna": self.loss_msi_rna.item(),
            "loss_rna_msi": self.loss_rna_msi.item(),
            "kl_rna_p": self.rna_kl_p.item(),
            "kl_rna_mod": self.rna_kl_mod.item(),
            "kl_rna_s": self.rna_kl_s.item(),
            "kl_msi_p": self.msi_kl_p.item(),
            "kl_msi_mod": self.msi_kl_mod.item(),
            "kl_msi_s": self.msi_kl_s.item(),
            "kl": self.kl.item(),
        }
        if self.batch_integration:
            vals["batch_integration"] = self.batch_integration.item()
            vals["batch_loss"] = self.batch_loss.item()
            if self.batch_integration_rna and self.batch_integration_msi:
                vals["batch_integration_rna"] = self.batch_integration_rna.item()
                vals["batch_integration_msi"] = self.batch_integration_msi.item()
        return vals

    def calculate_private(
        self,
        model_input: ModelInputT,
        model_output: ModelOutputT,
    ) -> None:
        rna_output = ModalityOutput.from_dict(model_output[Modality.rna.name])
        msi_output = ModalityOutput.from_dict(model_output[Modality.msi.name])

        (
            self.rna,
            self.rna_kl_p,
            self.rna_kl_mod,
            self.rna_kl_s,
        ) = self._loss_mod(model_input[Modality.rna.name], rna_output)
        (
            self.msi,
            self.msi_kl_p,
            self.msi_kl_mod,
            self.msi_kl_s,
        ) = self._loss_mod(model_input[Modality.msi.name], msi_output)

        self.private = (
            self.rna
            + self.msi
            + (self.rna_kl_p + self.rna_kl_s + self.msi_kl_p + self.msi_kl_s)
        )

    def _loss_mod(
        self,
        modality_input,
        modality_output: ModalityOutput,
    ):
        """
        Calculates private loss components (MSE and KL) for one modality.
        """
        kld_p = modality_output.latent_p.kld()
        kld_mod = modality_output.latent_mod.kld()
        kld_s = modality_output.latent_s.kld()

        x_pred = modality_output.x
        x_real = torch.squeeze(modality_input)

        loss = self.loss_function(x_pred, x_real, self.dropout)
        return (
            torch.mean(loss),
            self.beta * torch.mean(kld_p),
            self.beta * torch.mean(kld_mod),
            self.beta * torch.mean(kld_s),
        )

    def calculate_shared(
        self, model_input: ModelInputT, model_output: ModelOutputT
    ) -> None:
        rna_real = torch.squeeze(model_input[Modality.rna.name])
        msi_real = torch.squeeze(model_input[Modality.msi.name])

        self.rna_poe = torch.mean(
            self.loss_function(
                model_output["rna_poe"],
                rna_real,
                dropout=self.dropout,
            )
        )
        self.msi_poe = torch.mean(
            self.loss_function(
                model_output["msi_poe"],
                msi_real,
                dropout=self.dropout,
            )
        )

        self.loss_msi_rna = torch.mean(
            self.loss_function(
                model_output["msi_rna_loss"],
                msi_real,
                dropout=self.dropout,
            )
        )
        self.loss_rna_msi = torch.mean(
            self.loss_function(
                model_output["rna_msi_loss"],
                rna_real,
                dropout=self.dropout,
            )
        )

        self.kl = self.beta * torch.mean(Latent(**model_output["poe_latent"]).kld())

        self.shared = (
            self.loss_rna_msi
            + self.loss_msi_rna
            + self.rna_poe
            + self.msi_poe
            + self.kl
        )

    def calculate_batch_integration_loss(
        self,
        input: ModelInputT,
        output: ModelOutputT,
        on_privates: bool = False,
        perplexity: float = 30,
    ):
        """
        Tries to correct the POE latent space for batch effects with Harmony and calculates loss
        as LISI (Local Inverse Simpson Index) score.
        """
        latent = output["msi"]["latent_s"]["z"]

        n_neighbors = min(3 * perplexity, latent.shape[0] - 1)
        neighbors = nearest_neighbors(latent, n_neighbors)

        self.batch_loss = torch.nansum(
            1
            / compute_lisi(
                neighbors,
                input[BATCH_KEY],
                self.batch_num,
                perplexity,
            )
        )
        if not on_privates:
            self.batch_integration = self.gamma * self.batch_loss
            return

        self.batch_integration_msi = self._batch_integration_mod(
            input[BATCH_KEY],
            ModalityOutput.from_dict(output[Modality.msi.name]),
            perplexity,
        )
        self.batch_integration_rna = self._batch_integration_mod(
            input[BATCH_KEY],
            ModalityOutput.from_dict(output[Modality.rna.name]),
            perplexity,
        )

        self.batch_integration = self.gamma * self.batch_loss + self.delta * (
            self.batch_integration_msi + self.batch_integration_rna
        )

    def _batch_integration_mod(
        self,
        batch_id,
        modality_output: ModalityOutput,
        perplexity: float = 30,
    ):
        """
        Calculates INVERSE batch integration loss for one modality.
        """
        latent = modality_output.latent_p.z

        n_neighbors = min(3 * perplexity, latent.shape[0] - 1)
        neighbors = nearest_neighbors(latent, n_neighbors)

        return 1 / torch.nansum(
            1
            / compute_lisi(
                neighbors,
                batch_id,
                self.batch_num,
                perplexity,
            )
        )
