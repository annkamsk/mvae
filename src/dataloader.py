from typing import Any, Tuple

from src.utils import setup_mudata

from src.constants import BATCH_KEY, MOD_KEY
from src.types import Modality, ModelInput, ModelInputT, ObsModalityMembership
import torch
import numpy as np
from scipy import sparse
from mudata import MuData
from torch.utils.data import RandomSampler


class MultimodalDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        mdata: MuData,
        mod_ids: Any,
        extra_categorical_covs: Any,
    ):
        self.dataset = mdata
        self.mod_ids = torch.Tensor(mod_ids)
        self.extra_categorical_covs = torch.Tensor(extra_categorical_covs)
        self.indices = np.arange(mdata.n_obs)

    def __getitem__(self, index) -> ModelInputT:
        data1 = self._get_modality_data(index, Modality.rna)
        data2 = self._get_modality_data(index, Modality.msi)
        batch1 = self._get_batch_data(index, Modality.rna)
        batch2 = self._get_batch_data(index, Modality.msi)
        cat_covs = self.extra_categorical_covs[index]

        return ModelInput(
            torch.Tensor(data1),
            torch.Tensor(data2),
            self.mod_ids[index],
            torch.ByteTensor(batch1),
            torch.ByteTensor(batch2),
            cat_covs,
        ).to_dict()

    def _is_item_in_modality(self, index, modality: Modality) -> bool:
        if modality == Modality.rna:
            return self.mod_ids[index] in [
                ObsModalityMembership.ONLY_MOD1,
                ObsModalityMembership.PAIRED,
            ]
        else:
            return self.mod_ids[index] in [
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
            return self.dataset.mod[modality.name][idx, :].obs.loc[:, BATCH_KEY].values
        else:
            return [-1]

    def __len__(self):
        return self.dataset.shape[0]


def mudata_to_dataloader(
    mdata: MuData, batch_size: int, shuffle=False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    setup_mudata(mdata)
    dataset = MultimodalDataset(
        mdata,
        mdata.obs.mod_id.values,
        mdata.obs.extra_categorical_covs.values.astype(int),
    )
    sc_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=8,
    )

    paired_dataset = MultimodalDataset(
        mdata[mdata.obs.mod_id == ObsModalityMembership.PAIRED, :],
        mdata[mdata.obs.mod_id == ObsModalityMembership.PAIRED, :].obs.mod_id.values,
        mdata[
            mdata.obs.mod_id == ObsModalityMembership.PAIRED, :
        ].obs.extra_categorical_covs.values.astype(int),
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
