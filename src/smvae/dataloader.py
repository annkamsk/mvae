from typing import Any, Dict, List, Optional, Tuple

from src.constants import MOD_KEY

from src.smvae.types import ModelInputT, ModelInput

from src.types import Modality, ObsModalityMembership
from mudata import MuData
import numpy as np
import pandas as pd
from scipy import sparse
import torch


def split_into_train_test(
    mdata: MuData,
    train_size: float,
) -> Tuple[MuData, Optional[MuData]]:
    assert 0 < train_size <= 1, "train_size must be between 0 and 1"

    train_idx, test_idx = _split_to_balance_modalities(mdata, train_size)

    train_mdata = mdata.copy()[train_idx, :]
    test_mdata = mdata.copy()[test_idx, :] if len(test_idx) != 0 else None

    print(f"Train data size: {len(train_mdata.obs)}")
    if test_mdata:
        print(f"Test data size: {len(test_mdata.obs)}")
    return train_mdata, test_mdata


def _split_to_balance_modalities(mdata, train_size):
    """
    Splits mdata into train and test dataset so that cells represented only by first, only by second, or by both modalities are split proportionally.
    """
    idxs = np.arange(mdata.shape[0])
    train_idx = []
    test_idx = []
    for m in mdata.obs.mod_id.unique():
        idxs_m = idxs[mdata.obs.mod_id == m]
        n = len(idxs_m)
        n_train = int(n * train_size)
        perm_idx_m = np.random.permutation(n)
        train_idx += list(idxs_m[perm_idx_m[:n_train]])
        test_idx += list(idxs_m[perm_idx_m[n_train:]])
    return train_idx, test_idx


class BimodalDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        mdata: MuData,
        mod_ids: Any,
        batch_keys: Dict[str, Tuple[str, int]],
    ):
        self.dataset = mdata
        self.mod_ids = torch.Tensor(mod_ids)
        self.batch_keys = batch_keys
        self.indices = np.arange(mdata.n_obs)

    def __getitem__(self, index) -> ModelInputT:
        data1 = self._get_modality_data(index, Modality.mod1)
        data2 = self._get_modality_data(index, Modality.mod2)
        batch1 = self._get_batch_data(index, Modality.mod1)
        batch2 = self._get_batch_data(index, Modality.mod2)

        return ModelInput(
            torch.Tensor(data1),
            torch.Tensor(data2),
            self.mod_ids[index],
            batch1,
            batch2,
        ).to_dict()

    def _get_modality_data(self, index, modality: Modality):
        idx = self.dataset.obs.index[index]
        if self._is_item_in_modality(idx, modality):
            data = self.dataset.mod[modality.name][idx, :].X
        else:
            data = torch.empty((1, self.dataset.mod[modality.name].n_vars))
        return data.A if sparse.issparse(data) else data

    def _get_batch_data(self, index, modality: Modality):
        idx = self.dataset.obs.index[index]
        batch_categories = {}
        for batch_key, _ in self.batch_keys.values():
            if self._is_item_in_modality(index, modality):
                batch_categories[batch_key] = torch.ByteTensor(
                    self.dataset.mod[modality.name][idx, :].obs.loc[:, batch_key].values
                )
            else:
                batch_categories[batch_key] = torch.ByteTensor([-1])
        return batch_categories

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

    def __len__(self):
        return self.dataset.shape[0]


def mudata_to_dataloader(
    mdata: MuData, batch_keys: Dict, batch_size: int, shuffle=False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    setup_mudata(mdata)
    dataset = BimodalDataset(mdata, mdata.obs.mod_id.values, batch_keys)
    sc_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=8,
    )

    paired_dataset = BimodalDataset(
        mdata[mdata.obs.mod_id == ObsModalityMembership.PAIRED, :],
        mdata[mdata.obs.mod_id == ObsModalityMembership.PAIRED, :].obs.mod_id.values,
        batch_keys,
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


def setup_mudata(mdata: MuData, batch_keys: List[str]):
    batch_key_dict = {}
    for batch_key in batch_keys:
        key = f"batch_{batch_key}"
        mdata.obs[key] = pd.Categorical(pd.factorize(mdata.obs.loc[:, batch_key])[0])
        n_batch = len(mdata.obs[key].cat.categories)
        batch_key_dict[batch_key] = (key, n_batch)

        for modality in Modality:
            mdata.mod[modality.name].obs[key] = mdata.obs[key]

    if not MOD_KEY in mdata.obs.columns:
        mdata.obs[MOD_KEY] = mdata.obsm[Modality.mod1.name].astype(int) + (
            mdata.obsm[Modality.mod2.name].astype(int) * 2
        )
        mdata.mod[Modality.mod1.name].obs[MOD_KEY] = mdata[
            mdata.obsm[Modality.mod1.name] == True
        ].obs[MOD_KEY]
        mdata.mod[Modality.mod2.name].obs[MOD_KEY] = mdata[
            mdata.obsm[Modality.mod2.name] == True
        ].obs[MOD_KEY]
