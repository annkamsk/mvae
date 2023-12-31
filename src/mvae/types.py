from dataclasses import dataclass
from enum import IntEnum
from typing import Any, List, Optional, TypedDict

from src.latent import Latent
from src.latent import LatentT
from mudata import MuData
import pandas as pd
import torch


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

    @classmethod
    def from_int(cls, value: int) -> "ObsModalityMembership":
        if value == 1:
            return cls.ONLY_MOD1
        elif value == 2:
            return cls.ONLY_MOD2
        elif value == 3:
            return cls.PAIRED
        else:
            raise ValueError("Invalid value for ObsModalityMembership")

    @classmethod
    def from_mudata(cls, mdata: MuData, mod1_key: str, mod2_key: str) -> pd.Series:
        return pd.Series(
            mdata.obsm[mod1_key].astype(int) + (mdata.obsm[mod2_key].astype(int) * 2),
            index=mdata.obs.index,
        )


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
    batch_id: torch.ByteTensor

    def to_dict(self) -> ModelInputT:
        return {
            "rna": self.rna,
            "msi": self.msi,
            "batch_id": self.batch_id,
        }


ModalityInputT = TypedDict(
    "ModalityInputT",
    {
        "x": torch.Tensor,
        "batch_id": torch.ByteTensor,
    },
)


ModelOutputT = TypedDict(
    "ModelOutputT",
    {
        "rna": "ModalityOutputT",
        "msi": "ModalityOutputT",
        "poe_latent": LatentT,
        "rna_poe": torch.Tensor,
        "msi_poe": torch.Tensor,
        "rna_batch_free": torch.Tensor,
        "msi_batch_free": torch.Tensor,
        "rna_msi_loss": torch.Tensor,
        "msi_rna_loss": torch.Tensor,
    },
)

ModalityOutputT = TypedDict(
    "ModalityOutputT",
    {
        "x": torch.Tensor,
        "latent_p": LatentT,
        "latent_mod": LatentT,
        "latent_s": LatentT,
    },
)


@dataclass
class ModalityOutput:
    x: torch.Tensor
    latent_p: Latent
    latent_mod: Latent
    latent_s: Latent

    @classmethod
    def from_dict(cls, d: ModalityOutputT) -> "ModalityOutput":
        return cls(
            x=d["x"],
            latent_p=Latent(**d["latent_p"]),
            latent_mod=Latent(**d["latent_mod"]),
            latent_s=Latent(**d["latent_s"]),
        )

    def to_dict(self) -> ModalityOutputT:
        return {
            "x": self.x,
            "latent_p": self.latent_p.to_dict(),
            "latent_mod": self.latent_mod.to_dict(),
            "latent_s": self.latent_s.to_dict(),
        }
