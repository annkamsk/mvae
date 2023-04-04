from typing import List, Optional, Tuple
from src.vae.dataloader import setup_batch_key
from src.constants import BATCH_KEY
import scanpy as sc
import numpy as np
import torch
from src.types import TrainParams
from src.loss import compute_lisi
from src.vae.model import VAE
from anndata import AnnData
from src.vae.train import to_latent, predict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def umap(model: VAE, adata: AnnData, batch_keys: List[str], train_params=TrainParams()):
    emb = to_latent(model, adata, batch_keys, train_params)

    y = predict(model, adata, batch_keys, train_params)
    adata.layers["y"] = np.vstack(y)

    adata.obsm["z"] = np.vstack([x.numpy() for x in emb])
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, use_rep="z", n_neighbors=30)
    sc.tl.umap(adata)
    return adata.obsm["X_umap"]


def plot_embedding(
    model: VAE,
    adata: AnnData,
    keys: List[str] = ["tissue", "ann", "sample"],
    batch_keys: str = ["sample"],
    train_params: TrainParams = TrainParams(),
    leiden_res: float = 0.8,
) -> None:
    adata.obsm["X_vae"] = umap(model, adata, batch_keys, train_params)

    sc.tl.leiden(adata, resolution=leiden_res, key_added=f"r{leiden_res}")

    sc.pl.embedding(
        adata,
        "X_vae",
        color=keys,
        size=15,
        wspace=0.35,
    )


def classification_performance(
    model: VAE,
    adata: AnnData,
    key: str = "ann",
    train_params: TrainParams = TrainParams(),
) -> Tuple[float, RandomForestClassifier]:
    if "X_vae" not in adata.obsm.keys():
        adata.obsm["X_vae"] = umap(model, adata, train_params)

    (
        X_train_z_ann,
        X_test_z_ann,
        y_train_z_ann,
        y_test_z_ann,
    ) = train_test_split(
        adata.obsm["X_vae"], adata.obs[key], test_size=0.33, random_state=2137
    )

    rfc_z_shared_ann = RandomForestClassifier()
    rfc_z_shared_ann.fit(X_train_z_ann, y_train_z_ann)
    return rfc_z_shared_ann.score(X_test_z_ann, y_test_z_ann), rfc_z_shared_ann


def batch_integration(
    model: VAE,
    adata: AnnData,
    n_batch: int,
    batch_key: str = "sample",
    train_params: TrainParams = TrainParams(),
):
    setup_batch_key(adata, batch_key)
    emb = to_latent(model, adata, batch_key, train_params)
    X = torch.Tensor(np.vstack([x.numpy() for x in emb])).to(model.device)
    batch_id = torch.ByteTensor(adata.obs.loc[:, BATCH_KEY].values).to(model.device)
    return compute_lisi(X, batch_id, n_batch)


def plot_spatial(
    model: VAE,
    adata: AnnData,
    key: str = "ann",
    rfc: Optional[RandomForestClassifier] = None,
):
    if rfc is None:
        _, rfc = classification_performance(model, adata, key)

    adata.obs["predicted_ann"] = rfc.predict(adata.obsm["X_vae"])
    adata.obs["predicted_ann"] = adata.obs["predicted_ann"].astype("category")

    total_categories = set(adata.obs[key].cat.categories)

    for i, s in enumerate(adata.obs.loc[:, "sample"].unique()):
        mdata_tmp = adata[adata.obs.loc[:, "sample"] == s]
        subset_ann_cat = set(mdata_tmp.obs["ann"].cat.categories)
        mdata_tmp.obs["ann"] = mdata_tmp.obs["ann"].cat.add_categories(
            total_categories - subset_ann_cat
        )
        mdata_tmp.obs["ann"] = mdata_tmp.obs["ann"].cat.reorder_categories(
            total_categories
        )

        subset_predicted_cat = set(mdata_tmp.obs["predicted_ann"].cat.categories)
        mdata_tmp.obs["predicted_ann"] = mdata_tmp.obs[
            "predicted_ann"
        ].cat.add_categories(total_categories - subset_predicted_cat)
        mdata_tmp.obs["predicted_ann"] = mdata_tmp.obs[
            "predicted_ann"
        ].cat.reorder_categories(total_categories)

        sc.pl.spatial(
            mdata_tmp, color=[key, "predicted_ann"], spot_size=3, title=s, wspace=0.35
        )
