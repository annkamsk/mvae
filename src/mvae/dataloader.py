from typing import Tuple, Optional

from src.constants import BATCH_KEY, BATCH_N_KEY
from src.mvae.types import Modality, ModelInput, ModelInputT
import torch
import numpy as np
from scipy import sparse
from mudata import MuData
import pandas as pd


def split_into_train_test(
    mdata: MuData,
    train_size: float,
) -> Tuple[MuData, Optional[MuData]]:
    assert 0 < train_size <= 1, "train_size must be between 0 and 1"

    idxs = np.arange(mdata.shape[0])
    np.random.shuffle(idxs)

    n = int(train_size * len(idxs))
    train_idx = idxs[:n]
    test_idx = idxs[n:] if len(idxs) != n else []

    train_mdata = mdata.copy()[train_idx, :]
    test_mdata = mdata.copy()[test_idx, :] if len(test_idx) != 0 else None

    print(f"Train data size: {len(train_mdata.obs)}")
    if test_mdata:
        print(f"Test data size: {len(test_mdata.obs)}")
    return train_mdata, test_mdata


def setup_mudata(mdata: MuData, batch_key: str) -> int:
    mdata.obs[BATCH_KEY] = pd.Categorical(pd.factorize(mdata.obs.loc[:, batch_key])[0])
    n_batch = len(mdata.obs[BATCH_KEY].cat.categories)

    for modality in Modality:
        mdata.mod[modality.name].obs[BATCH_KEY] = mdata.obs[BATCH_KEY]

    mdata.uns[BATCH_N_KEY] = n_batch

    return n_batch


def mudata_to_dataloader(
    mdata: MuData, batch_size: int, shuffle: bool
) -> torch.utils.data.DataLoader:
    dataset = MultimodalDataset(mdata)
    sc_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=8,
    )
    return sc_dataloader


class MultimodalDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        mdata: MuData,
    ):
        self.dataset = mdata
        self.indices = np.arange(mdata.n_obs)

    def __getitem__(self, index) -> ModelInputT:
        data1 = self._get_modality_data(index, Modality.rna)
        data2 = self._get_modality_data(index, Modality.msi)
        batch_data = self._get_batch_data(index)

        return ModelInput(
            torch.Tensor(data1),
            torch.Tensor(data2),
            torch.ByteTensor(batch_data),
        ).to_dict()

    def _get_modality_data(self, index, modality: Modality):
        idx = self.dataset.obs.index[index]
        data = self.dataset.mod[modality.name][idx, :].X
        return data.A if sparse.issparse(data) else data

    def _get_batch_data(self, index):
        idx = self.dataset.obs.index[index]
        return self.dataset[idx, :].obs.loc[:, BATCH_KEY].values

    def __len__(self):
        return self.dataset.shape[0]
