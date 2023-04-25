from typing import Any, Callable, Dict, Tuple

from src.smvae.types import BATCH_KEY, ModalityOutput, ModelInputT, ModelOutputT

from src.loss import get_loss_fun, mmd, mse, compute_lisi, nearest_neighbors

import torch
from torch.autograd import Variable

from src.latent import Latent
from src.types import Modality


class LossCalculator:
    private = None
    shared = None
    batch_integration = None
    batch_integration_rna = None
    batch_integration_msi = None
    batch_loss = None
    mod1_mse = None
    mod2_mse = None
    mod1_kl_s = None
    mod2_kl_s = None
    mod1_kl_p = None
    mod2_kl_p = None

    summary_writer: Any
    beta: float
    gamma: float = None  # parameter scaling batch integration loss
    loss_function: Callable = mse
    dropout: bool = False
    batch_num: int = 0

    def __init__(
        self,
        summary_writer,
        beta,
        gamma=None,
        loss_function="mse",
        dropout=True,
        batch_num=0,
    ):
        self.summary_writer = summary_writer
        self.beta = beta
        if gamma:
            self.gamma = gamma
        self.loss_function = get_loss_fun(loss_function)
        self.dropout = dropout
        self.batch_num = batch_num

    @property
    def total_loss(self) -> torch.Tensor:
        total = self.private + self.shared
        if self.batch_integration:
            total += self.private
        return total

    @property
    def values(self) -> Dict[str, float]:
        vals = {
            "private": self.private.item(),
            "shared": self.shared.item(),
            "rna_mse": self.mod1_mse.item(),
            "msi_mse": self.mod2_mse.item(),
            "kl_rna_s": self.mod1_kl_s.item(),
            "kl_msi_s": self.mod2_kl_s.item(),
            "kl_rna_p": self.mod1_kl_p.item(),
            "kl_msi_p": self.mod2_kl_p.item(),
        }
        if self.batch_integration:
            vals["batch_integration"] = self.batch_integration.item()
            vals["batch_integration_msi"] = self.batch_integration_msi.item()
        return vals

    def calculate_private(self, model_input: ModelInputT, model_output: ModelOutputT):
        mod1_output = ModalityOutput.from_dict(model_output[Modality.rna.name])
        mod2_output = ModalityOutput.from_dict(model_output[Modality.msi.name])

        (
            self.mod1_mse,
            self.mod1_kl_p,
            self.mod1_kl_s,
        ) = self._loss_mod(model_input[Modality.rna.name], mod1_output)
        (
            self.mod2_mse,
            self.mod2_kl_p,
            self.mod2_kl_s,
        ) = self._loss_mod(model_input[Modality.msi.name], mod2_output)

        self.private = (
            self.mod1_mse
            + self.mod2_mse
            + (self.mod1_kl_p + self.mod1_kl_s + self.mod2_kl_p + self.mod2_kl_s)
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
        kld_s = modality_output.latent_s.kld()

        x_pred = modality_output.x
        x_real = torch.squeeze(modality_input)

        loss = self.loss_function(x_pred, x_real, self.dropout)
        return (
            torch.mean(loss),
            self.beta * torch.mean(kld_p),
            self.beta * torch.mean(kld_s),
        )

    def calculate_shared(self, model_input: ModelInputT, model_output: ModelOutputT):
        """
        Calculates Maximum Mean Discrepancy(MMD) between model_input and output.
        - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.
        """
        mod1_real = torch.squeeze(model_input[Modality.rna.name])
        mod2_real = torch.squeeze(model_input[Modality.msi.name])

        self.mod1_mod2_loss = torch.mean(
            self.loss_function(
                model_output["mod1_mod2_loss"],
                mod2_real,
                dropout=self.dropout,
            )
        )
        self.mod2_mod1_loss = torch.mean(
            self.loss_function(
                model_output["mod2_mod1_loss"],
                mod1_real,
                dropout=self.dropout,
            )
        )

        self.recovered_mod1_poe = torch.mean(
            self.loss_function(
                model_output["mod1_poe"],
                mod1_real,
                dropout=self.dropout,
            )
        )
        self.recovered_mod2_poe = torch.mean(
            self.loss_function(
                model_output["mod2_poe"],
                mod2_real,
                dropout=self.dropout,
            )
        )

        self.kl = self.beta * torch.mean(Latent(**model_output["poe_latent"]).kld())

        self.shared = (
            self.mod1_mod2_loss
            + self.mod2_mod1_loss
            + self.recovered_mod1_poe
            + self.recovered_mod2_poe
            + self.kl
        )

    def calculate_batch_integration_loss(
        self,
        model_input: ModelInputT,
        model_output: ModelOutputT,
        perplexity: float = 30,
    ):
        """
        Tries to correct the latent space for batch effects with Harmony and calculates loss
        as LISI (Local Inverse Simpson Index) score.
        """
        self.batch_integration_msi = self._batch_integration_mod(
            model_input[BATCH_KEY],
            ModalityOutput.from_dict(model_output["msi"]),
            perplexity,
        )
        self.batch_integration = (self.gamma or 1.0) * (self.batch_integration_msi)

        # latent = model_output["poe_latent"]["z"]

        # n_neighbors = min(3 * perplexity, latent.shape[0] - 1)
        # neighbors = nearest_neighbors(latent, n_neighbors)

        # self.batch_loss = torch.nansum(
        #     1
        #     / compute_lisi(
        #         neighbors,
        #         model_input[BATCH_KEY],
        #         self.batch_num,
        #         perplexity,
        #         self.summary_writer,
        #     )
        # )

        # # poe_corrected = harmonize(latent, batch_id)
        # self.batch_integration = (self.gamma or 1.0) * self.batch_loss

        if not self.gamma:
            self.gamma = 0.5 * self.private.item() / self.batch_integration.item()
            self.batch_integration = self.gamma * self.batch_integration

    def _batch_integration_mod(
        self,
        batch_id,
        modality_output: ModalityOutput,
        perplexity: float = 30,
    ):
        """
        Calculates batch integration loss for one modality.
        """
        latent = modality_output.latent_p.z

        n_neighbors = min(3 * perplexity, latent.shape[0] - 1)
        neighbors = nearest_neighbors(latent, n_neighbors)

        return torch.nansum(
            1
            / compute_lisi(
                neighbors,
                batch_id,
                self.batch_num,
                perplexity,
                self.summary_writer,
            )
        )
