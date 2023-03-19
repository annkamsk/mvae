from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TrainParams:
    learning_rate: float = 1e-4
    n_epochs: int = 500
    train_size: float = 1.0  # proportion of training data, between 0 and 1
    batch_size: int = 128
    shuffle: bool = True
    batch_split: Any = None
    leave_sample_out: Optional[str] = None  # name of sample to leave out in training
    train_patience: int = 20
    test_patience: int = 20
    rna_loss: str = "mse"
    msi_loss: str = "mse"
    dropout: bool = True
