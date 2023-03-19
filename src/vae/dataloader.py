from src.vae.types import VAEInputT
from anndata import AnnData
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from src.constants import BATCH_KEY, BATCH_N_KEY


class UniModalDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        adata: AnnData,
    ):
        self.dataset = adata
        self.indices = np.arange(adata.n_obs)

    def __getitem__(self, index) -> VAEInputT:
        idx = self.dataset.obs.index[index]
        data = self.dataset[idx, :].X
        batch = self.dataset[idx, :].obs.loc[:, BATCH_KEY].values

        if sparse.issparse(data):
            data = data.A

        return {
            "x": torch.Tensor(data),
            "batch_id": torch.ByteTensor(batch),
        }

    def __len__(self):
        return self.dataset.shape[0]


def adata_to_dataloader(adata: AnnData, batch_size: int, shuffle=False):
    setup_batch_key(adata)
    dataset = UniModalDataset(adata)
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=8,
    )


def setup_batch_key(adata: AnnData):
    if not BATCH_KEY in adata.obs.columns:
        adata.obs[BATCH_KEY] = pd.Categorical(
            pd.factorize(adata.obs.loc[:, "sample"])[0]
        )
        adata.uns[BATCH_N_KEY] = len(adata.obs[BATCH_KEY].cat.categories)
