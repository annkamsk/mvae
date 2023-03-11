from typing import List, Optional, Tuple

import pandas as pd

from src.constants import BATCH_KEY, BATCH_N_KEY, CAT_COVS_KEY, MOD_KEY
from src.types import Modality
import numpy as np
from mudata import MuData


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
