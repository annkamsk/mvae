from dataclasses import dataclass
from typing import TypedDict
from src.latent import LatentT

import torch


VAEInputT = TypedDict(
    "VAEInputT",
    {
        "x": torch.Tensor,
        "batch_id": torch.ByteTensor,
    },
)


@dataclass
class VAEInput:
    x: torch.Tensor
    batch_id: torch.ByteTensor

    def to_dict(self) -> VAEInputT:
        return {
            "x": self.x,
            "batch_id": self.batch_id,
        }


VAEOutputT = TypedDict(
    "VAEOutputT",
    {
        "x": torch.Tensor,
        "latent": LatentT,
    },
)
