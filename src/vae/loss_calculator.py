from typing import Any, Callable, Dict, Tuple

from src.constants import BATCH_KEY

from src.vae.types import VAEInputT, VAEOutputT

from src.loss import (
    compute_spatial_loss,
    get_loss_fun,
    mse,
    compute_lisi,
    nearest_neighbors,
)

import torch

from src.latent import Latent


class LossCalculator:
    private = None
    batch_integration = None
    batch_losses = {}
    spatial = None
    x = None
    kl = None

    summary_writer: Any
    beta: float
    gamma: float = 1.0  # scaling param for LISI
    loss_function: Callable = mse
    dropout: bool = False
    batch_key_dict: Dict[str, Tuple[str, int]] = {}

    def __init__(
        self,
        summary_writer,
        beta,
        gamma=1.0,
        loss_function="mse",
        dropout=True,
        batch_key_dict={},
    ):
        self.summary_writer = summary_writer
        self.beta = beta
        self.gamma = gamma
        self.loss_function = get_loss_fun(loss_function)
        self.dropout = dropout
        self.batch_key_dict = batch_key_dict

    @property
    def total_loss(self) -> torch.Tensor:
        total = self.private
        if self.batch_integration is not None:
            total += self.batch_integration
        if self.spatial is not None:
            total += self.spatial
        return total

    @property
    def values(self) -> Dict[str, float]:
        vals = {
            "private": self.private.item(),
            "x": self.x.item(),
            "kl": self.kl.item(),
        }
        if self.batch_integration:
            vals["batch_integration"] = self.batch_integration.item()
            for batch_key in self.batch_losses:
                vals[batch_key] = self.batch_losses[batch_key].item()
        if self.spatial:
            vals["spatial"] = self.spatial.item()
        return vals

    def calculate_private(self, model_input: VAEInputT, model_output: VAEOutputT):
        kld = Latent(**model_output["latent"]).kld()
        self.kl = self.beta * torch.mean(kld)

        x_pred = model_output["x"]
        x_real = torch.squeeze(model_input["x"])
        x_loss = self.loss_function(x_pred, x_real, self.dropout)
        self.x = torch.mean(x_loss)

        self.private = self.x + self.kl

    def calculate_batch_integration_loss(
        self,
        model_input: VAEInputT,
        model_output: VAEOutputT,
        perplexity: float = 30,
    ):
        """
        Calculates loss as inverse of LISI (Local Inverse Simpson Index) score.
        """
        latent = model_output["latent"]["z"]

        n_neighbors = min(3 * perplexity, latent.shape[0] - 1)
        distances, indices = nearest_neighbors(latent, n_neighbors)

        for batch_key, batch_n in self.batch_key_dict.values():
            batch_loss = torch.nansum(
                1
                / compute_lisi(
                    (distances, indices),
                    model_input[batch_key],
                    batch_n,
                    perplexity,
                    self.summary_writer,
                )
            )
            self.batch_losses[batch_key] = batch_loss

        # self.spatial = compute_spatial_loss(indices, model_input["neighbors"])

        # poe_corrected = harmonize(latent, batch_id)
        batch_integration = torch.stack(tuple(self.batch_losses.values())).sum()

        # if self.batch_integration_scale is None:
        #     self.batch_integration_scale = (
        #         self.gamma * self.private.item() / batch_integration.item()
        #     )

        self.batch_integration = self.gamma * batch_integration


class VAEBLossCalculator:
    msi_kl_p = None
    msi_kl_mod = None
    private = None
    x = None
    batch_integration = None

    summary_writer: Any
    beta: float
    gamma: float = 1.0  # scaling param for LISI
    loss_function: Callable = mse
    dropout: bool = False

    def __init__(
        self,
        beta,
        gamma=1.0,
        loss_function="mse",
        dropout=True,
    ):
        self.beta = beta
        self.gamma = gamma
        self.loss_function = get_loss_fun(loss_function)
        self.dropout = dropout

    @property
    def total_loss(self) -> torch.Tensor:
        total = self.private
        if self.batch_integration is not None:
            total += self.batch_integration
        return total

    @property
    def values(self) -> Dict[str, float]:
        vals = {
            "private": self.private.item(),
            "x": self.x.item(),
            "msi_kl_p": self.msi_kl_p.item(),
            "msi_kl_mod": self.msi_kl_mod.item(),
        }
        if self.batch_integration:
            vals["batch_integration"] = self.batch_integration.item()
        return vals

    def calculate_private(self, model_input: VAEInputT, model_output: VAEOutputT):
        self.msi_kl_p = (
            self.beta * 0.1 * torch.mean(Latent(**model_output["latent_b"]).kld())
        )
        self.msi_kl_mod = self.beta * torch.mean(
            Latent(**model_output["latent_mod"]).kld()
        )

        x_pred = model_output["x"]
        x_real = torch.squeeze(model_input["x"])

        self.x = torch.mean(self.loss_function(x_pred, x_real, self.dropout))
        self.private = self.x + self.msi_kl_p + self.msi_kl_mod

    def calculate_batch_integration_loss(
        self,
        model_input: VAEInputT,
        model_output: VAEOutputT,
        perplexity: float = 30,
    ):
        """
        Calculates loss as inverse of LISI (Local Inverse Simpson Index) score.
        """
        latent = model_output["latent_mod"]["z"]

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

        self.batch_integration = self.gamma * self.batch_integration
