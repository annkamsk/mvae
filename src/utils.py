from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_SIZE_GUIDANCE = {
    "compressedHistograms": 1,
    "images": 1,
    "scalars": 0,  # 0 means load all
    "histograms": 1,
}


class EarlyStopping:
    """
    Stops the training if the loss doesn't improve by more than delta after a given patience.
    """

    def __init__(
        self, patience: int = 7, verbose=False, delta: float = 0.01, mode="train"
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


def make_plot(data: pd.DataFrame, columns=[], title="") -> None:
    names = {
        "PoE_training/total_loss": "Total loss",
        "PoE_training/kl": "KL",
        "total_minus_bi": "Total loss - BI loss",
        "PoE_training/batch_integration": "BI loss",
        "PoE_training/rna_mse": "RNA MSE",
        "PoE_training/msi_mse": "MSI MSE",
        "PoE_training/private": "Private loss",
        "PoE_training/shared": "Shared loss",
        "PoE_training/kl_rna_p": "RNA KL p",
        "PoE_training/kl_rna_mod": "RNA KL mod",
        "PoE_training/kl_rna_s": "RNA KL s",
        "PoE_training/kl_msi_p": "MSI KL p",
        "PoE_training/kl_msi_mod": "MSI KL mod",
        "PoE_training/kl_msi_s": "MSI KL s",
        "PoE_training/recovered_rna_poe": "RNA recovered from PoE",
        "PoE_training/recovered_msi_poe": "MSI recovered from PoE",
        "PoE_training/loss_msi_rna": "Translation loss MSI -> RNA",
        "PoE_training/loss_rna_msi": "Translation loss RNA -> MSI",
    }
    n_col = 3
    n_row = int((len(columns) + (n_col - 1)) / n_col)
    fig_height = {1: 5, 2: 10, 3: 10, 4: 15}
    fig, axes = plt.subplots(n_row, n_col, sharex=True, figsize=(25, fig_height[n_row]))
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(wspace=0.3, top=0.8)

    if n_row < 2:
        for col in range(n_col):
            loss_type = columns[col]
            sns.lineplot(ax=axes[col], data=data[loss_type])
            axes[col].set_title(names[loss_type])
        return

    for row in range(n_row):
        for col in range(n_col):
            if row * n_col + col == len(columns):
                return
            loss_type = columns[row * n_col + col]
            sns.lineplot(ax=axes[row][col], data=data[loss_type])
            axes[row][col].set_title(names[loss_type])
            axes[row][col].set_ylabel("")


def get_loss_logs(model: str, mvae: bool = False):
    path = str(
        Path(__file__).parent.parent
        / "logs"
        / ("mvaevae_params" if mvae else "smvaevae_params")
        / model
    )
    event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]

    total_loss_events = event_acc.Scalars("PoE_training/total_loss")
    steps = list(map(lambda x: x.step, total_loss_events))
    runlog_data = pd.DataFrame({"step": steps})

    for tag in tags:
        event_list = event_acc.Scalars(tag)
        runlog_data[tag] = np.asarray(map(lambda x: x.value, event_list))

    return runlog_data
