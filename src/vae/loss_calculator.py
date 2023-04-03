from typing import Callable, Dict

from src.harmony import harmonize

from src.vae.types import VAEInputT, VAEOutputT

from src.loss import get_loss_fun, mse, compute_lisi

import torch

from src.latent import Latent


class LossCalculator:
    private = None
    batch_integration = None
    x = None
    kl = None

    beta: float
    loss_function: Callable = mse
    dropout: bool = False
    n_batch: int

    def __init__(self, beta, n_batch: int, loss_function="mse", dropout=True):
        self.beta = beta
        self.loss_function = get_loss_fun(loss_function)
        self.dropout = dropout
        self.n_batch = n_batch

    @property
    def total_loss(self) -> torch.Tensor:
        if self.batch_integration is None:
            return self.private
        return self.private + self.batch_integration

    @property
    def values(self) -> Dict[str, float]:
        vals = {
            "private": self.private.item(),
            "x": self.x.item(),
            "kl": self.kl.item(),
        }
        if self.batch_integration:
            vals["batch_integration"] = self.batch_integration.item()
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
        Tries to correct the latent space for batch effects with Harmony and calculates loss
        as LISI (Local Inverse Simpson Index) score.
        """
        batch_id = model_input["batch_id"]
        latent = model_output["latent"]["z"]

        # poe_corrected = harmonize(latent, batch_id)
        self.batch_integration = 0.01 * torch.nansum(
            1 / compute_lisi(latent, batch_id, self.n_batch, perplexity)
        )
