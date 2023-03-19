from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


class EarlyStopping:
    """
    Stops the training if the loss doesn't improve by more than delta after a given patience.
    """

    def __init__(
        self, patience: int = 7, verbose=False, delta: float = 0, mode="train"
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.mode = mode

    def __call__(self, val_loss: float, epoch: int):
        score = val_loss
        if self._has_score_improved(score):
            self.best_score = score
            self.counter = 0

            if self.verbose:
                print(
                    f"EarlyStopping (epoch: {epoch}): {self.mode} loss improved to {score}",
                    flush=True,
                )
            return

        self.counter += 1
        if self.counter >= self.patience:
            print(
                f"EarlyStopping (epoch: {epoch}): {self.mode} loss hasn't improved: {score}. Stopping.",
                flush=True,
            )
            self.early_stop = True

    def _has_score_improved(self, current_score) -> bool:
        return self.best_score is None or current_score < self.best_score - self.delta


def log_loss(writer, loss_values: Dict[str, float], iteration: int, train=True) -> None:
    for key, value in loss_values.items():
        if train:
            writer.add_scalar(f"PoE_training/{key}", value, iteration)
        else:
            writer.add_scalar(f"PoE_test/{key}", value, iteration)
