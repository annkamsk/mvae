from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import List, Tuple, TypedDict
import pandas as pd
import torch
import numpy as np
import src.constants as constants
from scipy import sparse
from mudata import MuData
from torch.utils.data import RandomSampler


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


MultimodalDatasetItemT = TypedDict(
    "MultimodalDatasetItemT",
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
class MultimodalDatasetItem:
    rna: torch.Tensor
    msi: torch.Tensor
    mod_id: int
    batch_id1: torch.ByteTensor
    batch_id2: torch.ByteTensor
    extra_categorical_covs: List[int]


class MultimodalDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        mdata: MuData,
        modality_membership: pd.Series,
        extra_categorical_covs: List[int] = [],
    ):
        self.dataset = mdata
        self.modality_membership = torch.Tensor(modality_membership)
        self.extra_categorical_covs = torch.Tensor(extra_categorical_covs)
        self.indices = np.arange(mdata.n_obs)

    def __getitem__(self, index) -> MultimodalDatasetItemT:
        data1 = self._get_modality_data(index, Modality.rna)
        data2 = self._get_modality_data(index, Modality.msi)
        batch1 = self._get_batch_data(index, Modality.rna)
        batch2 = self._get_batch_data(index, Modality.msi)
        cat_covs = self.extra_categorical_covs[index]

        return asdict(
            MultimodalDatasetItem(
                torch.Tensor(data1),
                torch.Tensor(data2),
                self.modality_membership[index],
                torch.ByteTensor(batch1),
                torch.ByteTensor(batch2),
                cat_covs,
            )
        )

    def _is_item_in_modality(self, index, modality: Modality) -> bool:
        if modality == Modality.rna:
            return self.modality_membership[index] in [
                ObsModalityMembership.ONLY_MOD1,
                ObsModalityMembership.PAIRED,
            ]
        else:
            return self.modality_membership[index] in [
                ObsModalityMembership.ONLY_MOD2,
                ObsModalityMembership.PAIRED,
            ]

    def _get_modality_data(self, index, modality: Modality):
        idx = self.dataset.obs.index[index]
        if self._is_item_in_modality(index, modality):
            data = self.dataset.mod[modality.name][idx, :].X
        else:
            data = torch.empty((1, self.dataset.mod[modality.name].n_vars))
        return data.A if sparse.issparse(data) else data

    def _get_batch_data(self, index, modality: Modality):
        idx = self.dataset.obs.index[index]
        if self._is_item_in_modality(index, modality):
            return (
                self.dataset.mod[modality.name][idx, :]
                .obs.loc[:, constants.BATCH_KEY]
                .values
            )
        else:
            return [-1]

    def __len__(self):
        return self.dataset.shape[0]


def mudata_to_dataloader(
    mdata: MuData, batch_size: int, shuffle=False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    modality_membership = ObsModalityMembership.from_mudata(
        mdata, Modality.rna.name, Modality.msi.name
    )
    dataset = MultimodalDataset(
        mdata,
        modality_membership,
        mdata.obs.extra_categorical_covs.values.astype(int),
    )
    sc_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=8,
    )

    paired_obs_idx = modality_membership[
        modality_membership == ObsModalityMembership.PAIRED
    ].index
    paired_obs = mdata[paired_obs_idx]
    paired_dataset = MultimodalDataset(
        paired_obs,
        modality_membership[paired_obs_idx].values,
        paired_obs.obs.extra_categorical_covs.values.astype(int),
    )
    sampler = RandomSampler(
        paired_dataset, replacement=True, num_samples=mdata.shape[0]
    )
    sc_dataloader_pairs = torch.utils.data.DataLoader(
        paired_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True,
    )
    return sc_dataloader, sc_dataloader_pairs
