from typing import Callable, Dict, Tuple

from src.harmony import harmonize

from src.vae.types import VAEInputT, VAEOutputT

from src.loss import get_loss_fun, mse, compute_lisi

import torch

from src.latent import Latent


class LossCalculator:
    private = None
    batch_integration = None
    batch_losses = {}
    x = None
    kl = None

    beta: float
    loss_function: Callable = mse
    dropout: bool = False
    batch_key_dict: Dict[str, Tuple[str, int]] = {}

    def __init__(self, beta, loss_function="mse", dropout=True, batch_key_dict={}):
        self.beta = beta
        self.loss_function = get_loss_fun(loss_function)
        self.dropout = dropout
        self.batch_key_dict = batch_key_dict

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
            for batch_key in self.batch_losses:
                vals[batch_key] = self.batch_losses[batch_key].item()
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
        latent = model_output["latent"]["z"]

        for batch_key, batch_n in self.batch_key_dict.values():
            batch_loss = torch.nansum(
                1
                / compute_lisi(
                    latent,
                    model_input[batch_key],
                    batch_n,
                    perplexity,
                )
            )
            if batch_key == "batch_batch":
                batch_loss = 0.1 * batch_loss
            elif batch_key == "batch_final_annotation":
                batch_loss = 10 / batch_loss
            self.batch_losses[batch_key] = batch_loss

        # poe_corrected = harmonize(latent, batch_id)
        self.batch_integration = torch.stack(tuple(self.batch_losses.values())).sum()
