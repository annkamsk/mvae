from typing import Dict, List, Tuple
from src.vae.types import VAEInputT
from anndata import AnnData
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import sklearn.neighbors


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

        if "spatial_net" in self.dataset.uns:
            neighbors = self.dataset.uns["spatial_net"].loc[idx, "target"]

        batch_categories = {}
        for batch_key, _ in self.batch_keys.values():
            batch_categories[batch_key] = torch.ByteTensor(
                self.dataset[idx, :].obs.loc[:, batch_key].values
            )

        if sparse.issparse(data):
            data = data.A

        return {
            "x": torch.Tensor(data),
            "neighbors": torch.Tensor(neighbors)
            if "spatial_net" in self.dataset.uns
            else None,
            **batch_categories,
        }

    def __len__(self):
        return self.dataset.shape[0]


def adata_to_dataloader(
    adata: AnnData, batch_keys: Dict, batch_size: int, shuffle=False
):
    if "spatial" in adata.obsm:
        setup_spatial_neighbor_network(adata)
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


def setup_spatial_neighbor_network(adata: AnnData, k_neigh=20):
    """
    Creates matrix with pairs of k_neigh spatial neighbors and their spatial distance.
    Neighbors are searched only within cells of the same sample.
    Saves matrix under adata.uns['spatial_net'].
    Adapted from: https://github.com/zhanglabtools/STAGATE/blob/main/STAGATE/utils.py
    """
    assert (
        "spatial" in adata.obsm.keys()
    ), "AnnData object must include spatial coordinates in adata.obsm['spatial']"

    spatial_net = []

    # mapping of cell names to integer indices in whole adata
    cell_map = dict(zip(adata.obs.index, range(len(adata.obs.index))))

    for sample in adata.obs["sample"].unique():
        sample_adata = adata[adata.obs["sample"] == sample, :]
        coor = pd.DataFrame(
            sample_adata.obsm["spatial"],
            index=sample_adata.obs.index,
            columns=["imagerow", "imagecol"],
        )
        # mapping of integer indices in sample adata to cell names
        index_map = dict(zip(range(len(coor)), coor.index))

        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_neigh + 1).fit(coor)
        distances_all, indices_all = nbrs.kneighbors(coor)

        # remove self
        distances, indices = distances_all[:, 1:], indices_all[:, 1:]

        # indices are integer indices of sample adata, need to convert to global adata indices
        neighbors_as_cells = np.apply_along_axis(
            lambda x: np.vectorize(cell_map.get)(np.vectorize(index_map.get)(x)),
            1,
            indices,
        )

        spatial_net.append(
            (
                coor.index,
                neighbors_as_cells,
                distances,
            )
        )

    # dataframe indexed by cells with lists of their spatial neighbors and distances
    adata.uns["spatial_net"] = (
        pd.DataFrame(spatial_net, columns=["source", "target", "distance"])
        .explode(["source", "target", "distance"], ignore_index=True)
        .set_index("source")
    )
