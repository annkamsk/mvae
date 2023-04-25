from typing import List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.mvae.train import to_latent
from src.mvae.model import MVAE
from mudata import MuData
from src.types import TrainParams
import scanpy as sc


def umap(
    model: MVAE,
    mdata: MuData,
    train_params: TrainParams = TrainParams(),
    poe_only: bool = True,
):
    poe, rna_p, msi_p, rna_mod, msi_mod, rna_s, msi_s = to_latent(
        model, mdata, train_params
    )
    if not poe_only:
        mdata.obsm["z1_s"] = np.vstack(rna_s)
        mdata.obsm["z2_s"] = np.vstack(msi_s)
        mdata.obsm["z1_p"] = np.vstack(rna_p)
        mdata.obsm["z2_p"] = np.vstack(msi_p)
        mdata.obsm["z1_mod"] = np.vstack(rna_mod)
        mdata.obsm["z2_mod"] = np.vstack(msi_mod)

        for layer in ["z1_s", "z2_s", "z1_p", "z2_p", "z1_mod", "z2_mod"]:
            sc.pp.neighbors(
                mdata, n_neighbors=5, use_rep=layer, key_added=f"neigh_{layer}"
            )
            sc.tl.umap(mdata, neighbors_key=f"neigh_{layer}")
            mdata.obsm[f"X_{layer}"] = mdata.obsm["X_umap"]

    mdata.obsm["z"] = np.vstack([x.numpy() for x in poe])

    sc.pp.neighbors(mdata, use_rep="z", n_neighbors=30)
    sc.tl.umap(mdata)
    return mdata.obsm["X_umap"]


def plot_embedding(
    model: MVAE,
    mdata: MuData,
    keys: List[str] = ["tissue", "new_ann", "sample"],
    train_params: TrainParams = TrainParams(),
    leiden_res: float = 0.8,
    poe_only: bool = True,
) -> None:
    mdata.obsm["X_mvae"] = umap(model, mdata, train_params, poe_only=poe_only)
    sc.tl.leiden(mdata, resolution=leiden_res, key_added=f"r{leiden_res}")
    sc.pl.embedding(
        mdata,
        "X_mvae",
        color=keys,
        size=15,
        wspace=0.35,
    )


def classification_performance(
    mdata: MuData,
    label_key: str = "new_ann",
    embed_key: str = "X_mvae",
    test_size: float = 0.33,
    loo: str = "",
) -> Tuple[float, RandomForestClassifier]:
    if len(loo):
        X_train = mdata[mdata.obs["sample"] != loo].obsm[embed_key]
        y_train = mdata[mdata.obs["sample"] != loo].obs[label_key]
        X_test = mdata[mdata.obs["sample"] == loo].obsm[embed_key]
        y_test = mdata[mdata.obs["sample"] == loo].obs[label_key]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            mdata.obsm[embed_key],
            mdata.obs[label_key],
            test_size=test_size,
            random_state=2137,
        )

    rfc_z_shared_ann = RandomForestClassifier()
    rfc_z_shared_ann.fit(X_train, y_train)
    return rfc_z_shared_ann.score(X_test, y_test), rfc_z_shared_ann
