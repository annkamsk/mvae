from typing import Dict, List, Tuple
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
        batch_keys: Dict[str, Tuple[str, int]],
    ):
        self.dataset = adata
        self.batch_keys = batch_keys
        self.indices = np.arange(adata.n_obs)

    def __getitem__(self, index) -> VAEInputT:
        idx = self.dataset.obs.index[index]
        data = self.dataset[idx, :].X

        batch_categories = {}
        for batch_key, _ in self.batch_keys.values():
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
    adata: AnnData, batch_keys: Dict, batch_size: int, shuffle=False
):
    dataset = UniModalDataset(adata, batch_keys)
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=8,
    )


def setup_batch_key(
    adata: AnnData, batch_keys: List[str]
) -> Dict[str, Tuple[str, int]]:
    # if not BATCH_KEY in adata.obs.columns:
    batch_key_dict = {}
    for batch_key in batch_keys:
        key = f"batch_{batch_key}"
        adata.obs[key] = pd.Categorical(pd.factorize(adata.obs.loc[:, batch_key])[0])
        n_batch = len(adata.obs[key].cat.categories)
        batch_key_dict[batch_key] = (key, n_batch)
    return batch_key_dict
