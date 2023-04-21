from typing import Any, Callable, Dict, Tuple

from src.loss import get_loss_fun, mse, compute_lisi

import torch

from src.latent import Latent


class LossCalculator:
    private = None
    shared = None
    batch_integration = None
    batch_losses = {}
    x = None
    kl = None

    summary_writer: Any
    beta: float
    gamma: float = None  # parameter scaling batch integration loss
    loss_function: Callable = mse
    dropout: bool = False
    batch_key_dict: Dict[str, Tuple[str, int]] = {}

    def __init__(
        self,
        summary_writer,
        beta,
        gamma=None,
        loss_function="mse",
        dropout=True,
        batch_key_dict={},
    ):
        self.summary_writer = summary_writer
        self.beta = beta
        if gamma:
            self.gamma = gamma
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
                    self.summary_writer,
                )
            )
            self.batch_losses[batch_key] = batch_loss

        # poe_corrected = harmonize(latent, batch_id)
        self.batch_integration = (self.gamma or 1.0) * torch.stack(
            tuple(self.batch_losses.values())
        ).sum()

        if not self.gamma:
            self.gamma = 0.5 * self.private.item() / self.batch_integration.item()
            self.batch_integration = self.gamma * self.batch_integration
