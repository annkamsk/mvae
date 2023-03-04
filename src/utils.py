from enum import Enum
from typing import List, Optional, Tuple
import warnings
import torch
import numpy as np
from .constants import _CONSTANTS
from scipy import sparse
from mudata import MuData
from torch.utils.data import RandomSampler


def _anndata_loader(mdata: MuData, batch_size: int, shuffle=False):
    """
    Load Anndata object into pytorch standard dataloader.
    Args:
        adata (AnnData): Scanpy Anndata object.
        batch_size (int): Cells per batch.
        shuffle (bool): Whether to shuffle data or not.
    iReturn:
        sc_dataloader (torch.DataLoader): Dataloader containing the data.
    """

    class multimodalDataset(torch.utils.data.dataset.Dataset):
        def __init__(self, _dataset):
            self.dataset = _dataset
            self.indices = np.arange(_dataset[0].n_obs)

        def __getitem__(self, index):
            idx = self.dataset[0].obs.index[index]
            mod_id = self.dataset[1][index]
            if mod_id == 1:
                if mod1_obsm:
                    data1 = (
                        self.dataset[0]
                        .mod[_CONSTANTS.MODALITY1_KEY][idx, :]
                        .obsm[mod1_obsm]
                    )
                else:
                    data1 = self.dataset[0].mod[_CONSTANTS.MODALITY1_KEY][idx, :].X
                if mod2_obsm:
                    data2 = torch.empty(
                        (
                            1,
                            self.dataset[0]
                            .mod[_CONSTANTS.MODALITY2_KEY]
                            .obsm[mod2_obsm]
                            .shape[1],
                        )
                    )
                else:
                    data2 = torch.empty(
                        (1, self.dataset[0].mod[_CONSTANTS.MODALITY2_KEY].n_vars)
                    )

                batch_id1 = (
                    self.dataset[0]
                    .mod[_CONSTANTS.MODALITY1_KEY][idx, :]
                    .obs.loc[:, _CONSTANTS.BATCH_KEY]
                    .values
                )
                batch_id2 = [-1]

            elif mod_id == 2:
                if mod2_obsm:
                    data2 = (
                        self.dataset[0]
                        .mod[_CONSTANTS.MODALITY2_KEY][idx, :]
                        .obsm[mod2_obsm]
                    )
                else:
                    data2 = self.dataset[0].mod[_CONSTANTS.MODALITY2_KEY][idx, :].X
                if mod1_obsm:
                    data1 = torch.empty(
                        (
                            1,
                            self.dataset[0]
                            .mod[_CONSTANTS.MODALITY1_KEY]
                            .obsm[mod1_obsm]
                            .shape[1],
                        )
                    )
                else:
                    data1 = torch.empty(
                        (1, self.dataset[0].mod[_CONSTANTS.MODALITY1_KEY].n_vars)
                    )

                batch_id1 = [-1]
                batch_id2 = (
                    self.dataset[0]
                    .mod[_CONSTANTS.MODALITY2_KEY][idx, :]
                    .obs.loc[:, _CONSTANTS.BATCH_KEY]
                    .values
                )

            elif mod_id == 3:
                if mod1_obsm:
                    data1 = (
                        self.dataset[0]
                        .mod[_CONSTANTS.MODALITY1_KEY][idx, :]
                        .obsm[mod1_obsm]
                    )
                else:
                    data1 = self.dataset[0].mod[_CONSTANTS.MODALITY1_KEY][idx, :].X
                if mod2_obsm:
                    data2 = (
                        self.dataset[0]
                        .mod[_CONSTANTS.MODALITY2_KEY][idx, :]
                        .obsm[mod2_obsm]
                    )
                else:
                    data2 = self.dataset[0].mod[_CONSTANTS.MODALITY2_KEY][idx, :].X

                batch_id1 = (
                    self.dataset[0]
                    .mod[_CONSTANTS.MODALITY1_KEY][idx, :]
                    .obs.loc[:, _CONSTANTS.BATCH_KEY]
                    .values
                )
                batch_id2 = (
                    self.dataset[0]
                    .mod[_CONSTANTS.MODALITY2_KEY][idx, :]
                    .obs.loc[:, _CONSTANTS.BATCH_KEY]
                    .values
                )

            if sparse.issparse(data1):
                data1 = data1.A
            if sparse.issparse(data2):
                data2 = data2.A
            cat_covs = self.dataset[2][index]

            return {
                _CONSTANTS.MODALITY1_KEY: torch.Tensor(data1),
                _CONSTANTS.MODALITY2_KEY: torch.Tensor(data2),
                "mod_id": mod_id,
                "batch_id1": torch.ByteTensor(batch_id1),
                "batch_id2": torch.ByteTensor(batch_id2),
                "extra_categorical_covs": cat_covs,
            }

        def __len__(self):
            return self.dataset[0].shape[0]

    # Encode modalities (mod1:1, mod2:2, paired:3)
    mdata.obs["mod_id"] = mdata.obsm[_CONSTANTS.MODALITY1_KEY].astype(int) + (
        mdata.obsm[_CONSTANTS.MODALITY2_KEY].astype(int) * 2
    )
    # mdata.obs['batch'] = mdata.obsm[_CONSTANTS.MODALITY1_KEY].astype(int) + (mdata.obsm[_CONSTANTS.MODALITY2_KEY].astype(int)*2)

    sc_dataloader = torch.utils.data.DataLoader(
        multimodalDataset(
            [
                mdata,
                torch.Tensor(mdata.obs["mod_id"].values),
                torch.Tensor(mdata.obs.extra_categorical_covs.values.astype(int)),
            ]
        ),
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=8,
    )

    paired_dataset = multimodalDataset(
        [
            mdata[mdata.obs.mod_id == 3, :],
            torch.Tensor(mdata[mdata.obs.mod_id == 3, :].obs["mod_id"].values),
            torch.Tensor(mdata[mdata.obs.mod_id == 3, :].obs.batch.values.astype(int)),
            torch.Tensor(
                mdata[
                    mdata.obs.mod_id == 3, :
                ].obs.extra_categorical_covs.values.astype(int)
            ),
        ]
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

    print(train_mdata)
    print(test_mdata)
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
        train_idx = np.where(mdata.obs.batch.isin(train_batches))[0]
        test_idx = np.where(mdata.obs.batch.isin(test_batches))[0]
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        return train_idx, test_idx

    return _split_to_balance_modalities(mdata, train_size)


def _split_to_balance_modalities(mdata, train_size):
    """
    Splits mdata into train and test dataset so that cells represented only by first, only by second, or by both modalities are split proportionally.
    """
    mdata.obs["mod_id"] = mdata.obsm[_CONSTANTS.MODALITY1_KEY].astype(int) + (
        mdata.obsm[_CONSTANTS.MODALITY2_KEY].astype(int) * 2
    )
    idxs = np.arange(mdata.shape[0])
    train_idx = []
    test_idx = []
    for m in mdata.obs.mod_id.unique():
        idxs_m = idxs[mdata.obs["mod_id"] == m]
        n = len(idxs_m)
        n_train = int(n * train_size)
        perm_idx_m = np.random.permutation(n)
        train_idx += list(idxs_m[perm_idx_m[:n_train]])
        test_idx += list(idxs_m[perm_idx_m[n_train:]])
    return train_idx, test_idx
