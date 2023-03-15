from src.types import ModelInputT, ModelOutputT
from mudata import MuData
from anndata import AnnData
import scanpy as sc
from harmony import harmonize
from src.constants import BATCH_KEY
from src.utils import setup_mudata


def harmony_correct(model_input: ModelInputT, model_output: ModelOutputT):
    rna = model_input["rna"].squeeze().detach().cpu().numpy()
    msi = model_input["msi"].squeeze().detach().cpu().numpy()
    poe = model_output["poe_latent"]["z"].detach().cpu()

    mdata = MuData({"rna": AnnData(rna), "msi": AnnData(msi)})
    mdata.obsm["z_poe"] = poe
    sc.pp.neighbors(mdata, use_rep="z_poe", n_neighbors=10)
    sc.tl.umap(mdata)
    mdata.obsm["X_mvae"] = mdata.obsm["X_umap"]
    setup_mudata(mdata)
    Z = harmonize(mdata.obsm["X_mvae"], mdata.obs, batch_key=BATCH_KEY)
    mdata.obsm["X_harmony"] = Z
    return mdata
