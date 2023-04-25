from typing import Optional, Tuple

from src.smvae.types import BATCH_KEY, ModelInputT, ModelInput

from src.types import Modality
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


class BimodalDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        mdata: MuData,
        batch_num: int,
    ):
        self.dataset = mdata
        self.batch_num = batch_num
        self.indices = np.arange(mdata.n_obs)

    def __getitem__(self, index) -> ModelInputT:
        data1 = self._get_modality_data(index, Modality.rna)
        data2 = self._get_modality_data(index, Modality.msi)
        batch_data = self._get_batch_data(index)

        return ModelInput(
            torch.Tensor(data1),
            torch.Tensor(data2),
            batch_data,
        ).to_dict()

    def _get_modality_data(self, index, modality: Modality):
        idx = self.dataset.obs.index[index]
        data = self.dataset.mod[modality.name][idx, :].X
        return data.A if sparse.issparse(data) else data

    def _get_batch_data(self, index):
        idx = self.dataset.obs.index[index]
        return torch.ByteTensor(self.dataset[idx, :].obs.loc[:, BATCH_KEY].values)

    def __len__(self):
        return self.dataset.shape[0]


def mudata_to_dataloader(
    mdata: MuData,
    batch_num: int,
    batch_size: int,
    shuffle=False,
) -> torch.utils.data.DataLoader:
    dataset = BimodalDataset(mdata, batch_num)
    sc_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=8,
    )

    return sc_dataloader


def setup_mudata(mdata: MuData, batch_key: str) -> int:
    mdata.obs[BATCH_KEY] = pd.Categorical(pd.factorize(mdata.obs.loc[:, batch_key])[0])
    n_batch = len(mdata.obs[BATCH_KEY].cat.categories)

    for modality in Modality:
        mdata.mod[modality.name].obs[BATCH_KEY] = mdata.obs[BATCH_KEY]

    return n_batch
