from typing import List, Optional, Tuple
import scanpy as sc
import numpy as np
from mudata import MuData


def plot_emb_mod(pred, mod, batch):
    pred = np.concatenate(pred)
    mod = np.concatenate(mod).flatten()
    batch = np.concatenate(batch).flatten()

    ad = sc.AnnData(pred, obs=pd.DataFrame({"mod": mod, "batch": batch}))
    ad.obs["mod"] = ad.obs["mod"].astype("category")
    ad.obs["batch"] = ad.obs["batch"].astype("category")
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)
    sc.pl.umap(ad, color=["mod", "batch"])


def plot_emb_batch(pred, batch):
    pred = np.concatenate(pred)
    batch = np.concatenate(batch).flatten()
    ad = sc.AnnData(pred, obs=pd.DataFrame({"batch": batch}))
    ad.obs["batch"] = ad.obs["batch"].astype("category")
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)
    sc.pl.umap(ad, color=["batch"])


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
