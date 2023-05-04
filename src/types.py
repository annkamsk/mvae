from dataclasses import dataclass
import datetime
from enum import IntEnum
from typing import Any, Optional


class Modality(IntEnum):
    rna = 1
    msi = 2


class ObsModalityMembership(IntEnum):
    """
    Represents membership status of an observation across modalities.
    """

    ONLY_MOD1 = 1
    ONLY_MOD2 = 2
    PAIRED = 3


@dataclass
class TrainParams:
    learning_rate: float = 1e-4
    n_epochs: int = 500
    train_size: float = 1.0  # proportion of training data, between 0 and 1
    batch_size: int = 128
    shuffle: bool = True
    batch_split: Any = None
    leave_sample_out: Optional[str] = None  # name of sample to leave out in training
    train_patience: int = 10
    test_patience: int = 10
    rna_loss: str = "mse"
    msi_loss: str = "mse"
    add_lisi_loss: bool = True
    inverse_lisi_on_private: bool = False
    dropout: bool = True
    params_file: str = ""

    def get_params_file(self) -> str:
        return f"vae_params/{self.params_file}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pt"
