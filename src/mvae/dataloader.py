from typing import Any, Tuple, List, Optional

from src.constants import BATCH_KEY, MOD_KEY, BATCH_N_KEY, CAT_COVS_KEY
from src.mvae.types import Modality, ModelInput, ModelInputT, ObsModalityMembership
import torch
import numpy as np
from scipy import sparse
from mudata import MuData
from torch.utils.data import RandomSampler
import pandas as pd


def split_into_train_test(
    mdata: MuData,
    train_size: float,
    sample: Optional[str] = None,
    batch_split: Optional[Tuple[List[str], List[str]]] = None,
) -> Tuple[MuData, Optional[MuData]]:
    """
    Splits MuData object into a training and test set. Test proportion is 1-train_size.
    If sample is set to a sample name then leave-sample-out method will be used.
    Else if batch_split is set then batch split method will be used.
    Else random split method will be used.
    """
    assert 0 < train_size <= 1, "train_size must be between 0 and 1"

    train_idx, test_idx = _split_indices(mdata, train_size, sample, batch_split)

    train_mdata = mdata.copy()[train_idx, :]
    test_mdata = mdata.copy()[test_idx, :] if len(test_idx) != 0 else None

    print(f"Train data size: {len(train_mdata.obs)}")
    if test_mdata:
        print(f"Test data size: {len(test_mdata.obs)}")
    return train_mdata, test_mdata


def _split_indices(
    mdata: MuData,
    train_size: float,
    sample: Optional[str],
    batch_split: Optional[Tuple[List[str], List[str]]],
):
    if sample is not None:
        train_idx = mdata.obs["sample"] != sample
        test_idx = mdata.obs["sample"] == sample
        return train_idx, test_idx

    if batch_split is not None:
        train_batches, test_batches = batch_split
        train_idx = np.where(mdata.obs.batch_id.isin(train_batches))[0]
        test_idx = np.where(mdata.obs.batch_id.isin(test_batches))[0]
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        return train_idx, test_idx

    return _split_to_balance_modalities(mdata, train_size)


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


def setup_mudata(mdata: MuData):
    if not CAT_COVS_KEY in mdata.obs.columns:
        mdata.obs[CAT_COVS_KEY] = 0

    if not BATCH_KEY in mdata.obs.columns:
        mdata.obs[BATCH_KEY] = pd.Categorical(
            pd.factorize(mdata.obs.loc[:, "sample"])[0]
        )

        for modality in Modality:
            mdata.mod[modality.name].obs[BATCH_KEY] = pd.Categorical(
                pd.factorize(mdata.obs.loc[:, "sample"])[0]
            )
            mdata.mod[modality.name].uns[BATCH_N_KEY] = len(
                mdata.mod[modality.name].obs[BATCH_KEY].cat.categories
            )

    if not MOD_KEY in mdata.obs.columns:
        mdata.obs[MOD_KEY] = mdata.obsm[Modality.rna.name].astype(int) + (
            mdata.obsm[Modality.msi.name].astype(int) * 2
        )
        mdata.mod[Modality.rna.name].obs[MOD_KEY] = mdata[
            mdata.obsm[Modality.rna.name] == True
        ].obs[MOD_KEY]
        mdata.mod[Modality.msi.name].obs[MOD_KEY] = mdata[
            mdata.obsm[Modality.msi.name] == True
        ].obs[MOD_KEY]


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
