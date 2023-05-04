from typing import Any, Callable, Dict

from src.smvae.types import BATCH_KEY, ModalityOutput, ModelInputT, ModelOutputT

from src.loss import get_loss_fun, mse, compute_lisi, nearest_neighbors

import torch

from src.latent import Latent
from src.types import Modality


class LossCalculator:
    private = None
    shared = None
    batch_integration = None
    batch_integration_scale = None
    batch_integration_rna = None
    batch_integration_msi = None
    batch_loss = None
    mod1_mse = None
    mod2_mse = None
    mod1_poe_mse = None
    mod2_poe_mse = None
    mod2_mod1_loss = None
    mod1_mod2_loss = None
    mod1_kl_s = None
    mod2_kl_s = None
    mod1_kl_p = None
    mod2_kl_p = None
    kl = None

    summary_writer: Any
    beta: float
    gamma: float = 1.0  # proportion between LISI los and private loss
    delta: float = 0.1
    loss_function: Callable = mse
    dropout: bool = False
    batch_num: int = 0

    def __init__(
        self,
        summary_writer,
        beta,
        gamma=1.0,
        delta=1.0,
        loss_function="mse",
        dropout=True,
        batch_num=0,
    ):
        self.summary_writer = summary_writer
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
            "rna_mse": self.mod1_mse.item(),
            "msi_mse": self.mod2_mse.item(),
            "recovered_rna_poe": self.mod1_poe_mse.item(),
            "recovered_msi_poe": self.mod2_poe_mse.item(),
            "loss_msi_rna": self.mod2_mod1_loss.item(),
            "loss_rna_msi": self.mod1_mod2_loss.item(),
            "kl": self.kl.item(),
            "kl_rna_p": self.mod1_kl_p.item(),
            "kl_msi_p": self.mod2_kl_p.item(),
            "kl_rna_s": self.mod1_kl_s.item(),
            "kl_msi_s": self.mod2_kl_s.item(),
        }
        if self.batch_integration:
            vals["batch_integration"] = self.batch_integration.item()
            vals["batch_loss"] = self.batch_loss.item()
            if self.batch_integration_rna and self.batch_integration_msi:
                vals["batch_integration_rna"] = self.batch_integration_rna.item()
                vals["batch_integration_msi"] = self.batch_integration_msi.item()
        return vals

    def calculate_private(self, model_input: ModelInputT, model_output: ModelOutputT):
        rna_output = ModalityOutput.from_dict(model_output[Modality.rna.name])
        msi_output = ModalityOutput.from_dict(model_output[Modality.msi.name])

        self.mod1_mse, self.mod1_kl_p, self.mod1_kl_s = self._loss_mod(
            torch.squeeze(model_input[Modality.rna.name]), rna_output
        )
        self.mod2_mse, self.mod2_kl_p, self.mod2_kl_s = self._loss_mod(
            torch.squeeze(model_input[Modality.msi.name]), msi_output
        )

        self.private = (
            self.mod1_mse
            + self.mod2_mse
            + self.mod1_kl_s
            + self.mod2_kl_s
            + self.mod1_kl_p
            + self.mod2_kl_p
        )

    def _loss_mod(
        self,
        modality_input,
        modality_output: ModalityOutput,
    ):
        """
        Calculates private loss components (MSE and KL) for one modality.
        """
        kld_s = modality_output.latent_s.kld()
        kld_p = modality_output.latent_p.kld()

        x_pred = modality_output.x
        x_real = torch.squeeze(modality_input)

        loss = self.loss_function(x_pred, x_real, self.dropout)
        return (
            torch.mean(loss),
            self.beta * torch.mean(kld_p),
            self.beta * torch.mean(kld_s),
        )

    def calculate_shared(self, model_input: ModelInputT, model_output: ModelOutputT):
        mod1_poe_output = model_output["rna_poe"]
        mod2_poe_output = model_output["msi_poe"]

        rna_real = torch.squeeze(model_input[Modality.rna.name])
        msi_real = torch.squeeze(model_input[Modality.msi.name])

        self.mod1_poe_mse = torch.mean(
            self.loss_function(mod1_poe_output, rna_real, self.dropout)
        )
        self.mod2_poe_mse = torch.mean(
            self.loss_function(mod2_poe_output, msi_real, self.dropout)
        )

        self.mod2_mod1_loss = torch.mean(
            self.loss_function(model_output["msi_rna_loss"], msi_real, self.dropout)
        )
        self.mod1_mod2_loss = torch.mean(
            self.loss_function(model_output["rna_msi_loss"], rna_real, self.dropout)
        )

        self.kl = self.beta * torch.mean(Latent(**model_output["poe_latent"]).kld())

        self.shared = (
            self.mod1_poe_mse
            + self.mod2_poe_mse
            + self.mod1_mod2_loss
            + self.mod2_mod1_loss
            + self.kl
        )

    def calculate_batch_integration_loss(
        self,
        model_input: ModelInputT,
        model_output: ModelOutputT,
        on_privates: bool = False,
        perplexity: float = 30,
    ):
        """
        Tries to correct the latent space for batch effects with Harmony and calculates loss
        as LISI (Local Inverse Simpson Index) score.
        """
        latent = model_output["poe_latent"]["z"]

        n_neighbors = min(3 * perplexity, latent.shape[0] - 1)
        neighbors = nearest_neighbors(latent, n_neighbors)

        self.batch_loss = torch.nansum(
            1
            / compute_lisi(
                neighbors,
                model_input[BATCH_KEY],
                self.batch_num,
                perplexity,
                self.summary_writer,
            )
        )
        if not on_privates:
            self.batch_integration = self.gamma * self.batch_loss
            return

        self.batch_integration_msi = self._batch_integration_mod(
            model_input[BATCH_KEY],
            ModalityOutput.from_dict(model_output[Modality.msi.name]),
            perplexity,
        )
        self.batch_integration_rna = self._batch_integration_mod(
            model_input[BATCH_KEY],
            ModalityOutput.from_dict(model_output[Modality.rna.name]),
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
                self.summary_writer,
            )
        )
