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
from sklearn.neighbors import NearestNeighbors


def umap(model: VAE, adata: AnnData, train_params=TrainParams()):
    emb = to_latent(model, adata, train_params)

    y = predict(model, adata, train_params)
    adata.layers["y"] = np.vstack(y)

    adata.obsm["z"] = np.vstack([x.numpy() for x in emb])

    sc.pp.neighbors(adata, use_rep="z", n_neighbors=10)
    sc.tl.umap(adata)
    return adata.obsm["X_umap"]


def plot_embedding(
    model: VAE,
    adata: AnnData,
    train_params: TrainParams = TrainParams(),
    leiden_res: float = 0.8,
) -> None:
    adata.obsm["X_vae"] = umap(model, adata, train_params)

    sc.tl.leiden(adata, resolution=leiden_res, key_added=f"r{leiden_res}")

    sc.pl.embedding(
        adata,
        "X_vae",
        color=["tissue", f"r{leiden_res}", "ann", "sample"],
        size=15,
        wspace=0.35,
    )


def classification_performance(
    model: VAE, adata: AnnData, train_params: TrainParams = TrainParams()
) -> float:
    if "X_vae" not in adata.obsm.keys():
        adata.obsm["X_vae"] = umap(model, adata, train_params)

    (
        X_train_z_ann,
        X_test_z_ann,
        y_train_z_ann,
        y_test_z_ann,
    ) = train_test_split(
        adata.obsm["X_vae"], adata.obs["ann"], test_size=0.33, random_state=2137
    )

    rfc_z_shared_ann = RandomForestClassifier()
    rfc_z_shared_ann.fit(X_train_z_ann, y_train_z_ann)
    return rfc_z_shared_ann.score(X_test_z_ann, y_test_z_ann)


def batch_integration(
    model: VAE, adata: AnnData, train_params: TrainParams = TrainParams()
):
    setup_batch_key(adata)
    emb = to_latent(model, adata, train_params)
    X = torch.Tensor(np.vstack([x.numpy() for x in emb])).to(model.device)
    batch_id = torch.ByteTensor(adata.obs.loc[:, BATCH_KEY].values).to(model.device)

    return compute_lisi(X, batch_id)
