from dataclasses import dataclass
from typing import Any, TypedDict
from src.latent import Latent, LatentT

import torch


BATCH_KEY = "batch_id"


ModelInputT = TypedDict(
    "ModelInputT",
    {
        "rna": torch.Tensor,
        "msi": torch.Tensor,
        "batch_id": torch.ByteTensor,
    },
)


@dataclass
class ModelInput:
    rna: torch.Tensor
    msi: torch.Tensor
    batch_sample: torch.ByteTensor

    def to_dict(self) -> ModelInputT:
        return {
            "rna": self.rna,
            "msi": self.msi,
            "batch_id": self.batch_sample,
        }


ModalityInputT = TypedDict(
    "ModalityInputT",
    {
        "x": torch.Tensor,
        "mod_id": Any,
        "batch_id": torch.ByteTensor,
    },
)


ModelOutputT = TypedDict(
    "ModelOutputT",
    {
        "mod1": "ModalityOutputT",
        "mod2": "ModalityOutputT",
        "poe_latent": LatentT,
        "mod1_poe": torch.Tensor,
        "mod2_poe": torch.Tensor,
        "mod1_mod2_loss": torch.Tensor,
        "mod2_mod1_loss": torch.Tensor,
    },
)

ModalityOutputT = TypedDict(
    "ModalityOutputT",
    {
        "x": torch.Tensor,
        "latent_p": LatentT,
        "latent_s": LatentT,
    },
)


@dataclass
class ModalityOutput:
    x: torch.Tensor
    latent_p: Latent
    latent_s: Latent

    @classmethod
    def from_dict(cls, d: ModalityOutputT) -> "ModalityOutput":
        return cls(
            x=d["x"],
            latent_p=Latent(**d["latent_p"]),
            latent_s=Latent(**d["latent_s"]),
        )

    def to_dict(self) -> ModalityOutputT:
        return {
            "x": self.x,
            "latent_p": self.latent_p.to_dict(),
            "latent_s": self.latent_s.to_dict(),
        }
