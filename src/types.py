from dataclasses import dataclass
from enum import IntEnum
from typing import List, TypedDict

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
        "mod_id": int,
        "batch_id1": torch.ByteTensor,
        "batch_id2": torch.ByteTensor,
        "extra_categorical_covs": torch.Tensor,
    },
)


@dataclass
class ModelInput:
    rna: torch.Tensor
    msi: torch.Tensor
    mod_id: int
    batch_id1: torch.ByteTensor
    batch_id2: torch.ByteTensor
    extra_categorical_covs: List[int]


ModelOutputT = TypedDict(
    "ModelOutputT",
    {
        "rna": "ModalityOutputT",
        "msi": "ModalityOutputT",
        "poe_latent": LatentT,
        "rna_msi_loss": torch.Tensor,
        "msi_rna_loss": torch.Tensor,
    },
)

ModalityOutputT = TypedDict(
    "ModalityOutputT",
    {
        "y": torch.Tensor,
        "y_batch_only": torch.Tensor,
        "y_poe": torch.Tensor,
        "y_batch_free": torch.Tensor,
        "latent_p": LatentT,
        "latent_mod": LatentT,
        "latent_s": LatentT,
    },
)


@dataclass
class ModalityOutput:
    y: torch.Tensor
    y_batch_only: torch.Tensor
    y_poe: torch.Tensor
    y_batch_free: torch.Tensor
    latent_p: Latent
    latent_mod: Latent
    latent_s: Latent

    @classmethod
    def from_dict(cls, d: ModalityOutputT) -> "ModalityOutput":
        return cls(
            y=d["y"],
            y_batch_only=d["y_batch_only"],
            y_poe=d["y_poe"],
            y_batch_free=d["y_batch_free"],
            latent_p=Latent(**d["latent_p"]),
            latent_mod=Latent(**d["latent_mod"]),
            latent_s=Latent(**d["latent_s"]),
        )
