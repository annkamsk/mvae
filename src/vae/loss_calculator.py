from typing import Callable, Dict

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

    def __init__(self, beta, loss_function="mse", dropout=True):
        self.beta = beta
        self.loss_function = get_loss_fun(loss_function)
        self.dropout = dropout

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
        poe = model_output["latent"]["z"]
        # poe_corrected = harmonize(poe, batch_id, device_type=device)
        self.batch_integration = 1 / torch.nanmean(compute_lisi(poe, batch_id, perplexity))
