from typing import List
from src.vae.types import VAEInputT
from anndata import AnnData
import numpy as np
import pandas as pd
from scipy import sparse
import torch


BATCH_KEYS = {}


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

        batch_categories = {}
        for batch_key, _ in BATCH_KEYS.values():
            batch_categories[batch_key] = torch.ByteTensor(
                self.dataset[idx, :].obs.loc[:, batch_key].values
            )

        if sparse.issparse(data):
            data = data.A

        return {
            "x": torch.Tensor(data),
            **batch_categories,
        }

    def __len__(self):
        return self.dataset.shape[0]


def adata_to_dataloader(
    adata: AnnData, batch_size: int, batch_keys: List[str], shuffle=False
):
    setup_batch_key(adata, batch_keys)
    dataset = UniModalDataset(adata)
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=8,
    )


def setup_batch_key(adata: AnnData, batch_keys: List[str]):
    # if not BATCH_KEY in adata.obs.columns:
    BATCH_KEYS = {}
    for batch_key in batch_keys:
        key = f"batch_{batch_key}"
        adata.obs[key] = pd.Categorical(pd.factorize(adata.obs.loc[:, batch_key])[0])
        n_batch = len(adata.obs[key].cat.categories)
        BATCH_KEYS[batch_key] = (key, n_batch)
