from dataclasses import dataclass
import datetime
from typing import Dict

from src.dataloader import mudata_to_dataloader

from src.utils import split_into_train_test

from src.loss import Loss

from src.types import ModalityOutput, ModelInputT, ModelOutputT

import numpy as np
from tqdm import tqdm
from src.model import MVAE
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim


@dataclass
class TrainParams:
    """
    Parameters
    ----------
    learning_rate
        learning rate
    n_epochs
        number of epochs to train model
    train_size
        a number between 0 and 1 to indicate the proportion of training data. Test size is set to 1-train_size
    batch_size
        number of samples per batch
    shuffle
        whether to shuffle samples or not
    leave_sample_out
        str: name of sample to leave out in training
    """

    learning_rate = 1e-4
    n_epochs = 500
    train_size = 1.0
    batch_size = 128
    batch_split = None
    shuffle = True
    leave_sample_out = None
    train_patience = 20
    test_patience = 20
    rna_loss = "mse"
    msi_loss = "bce"


class EarlyStopping:
    """
    Stops the training if the loss doesn't improve by more than delta after a given patience.
    """

    def __init__(
        self, patience: int = 7, verbose=False, delta: float = 0, mode="train"
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.mode = mode

    def __call__(self, val_loss):
        score = -val_loss
        if self._has_score_improved(score):
            self.best_score = score
            self.counter = 0

            if self.verbose:
                print(f"EarlyStopping: {self.mode} loss improved to {score}")
            return

        self.counter += 1
        if self.counter >= self.patience:
            print(
                f"EarlyStopping: {self.mode} loss hasn't improved: {score}. Stopping."
            )
            self.early_stop = True

    def _has_score_improved(self, current_score) -> bool:
        return self.best_score is None or current_score > self.best_score - self.delta


def log_loss(writer, loss_values: Dict[str, float], iteration: int, train=True) -> None:
    for key, value in loss_values.items():
        if train:
            writer.add_scalar(f"PoE_training/{key}", value, iteration)
        else:
            writer.add_scalar(f"PoE_test/{key}", value, iteration)


def extract_latent(model_output: ModelOutputT):
    rna_output = ModalityOutput.from_dict(model_output["rna"])
    msi_output = ModalityOutput.from_dict(model_output["msi"])
    return {
        "poe_latent": model_output["poe_latent"]["z"].cpu().numpy(),
        "rna_p": rna_output.latent_p.z.cpu().numpy(),
        "rna_mod": rna_output.latent_mod.z.cpu().numpy(),
        "rna_s": rna_output.latent_s.z.cpu().numpy(),
        "msi_p": msi_output.latent_p.z.cpu().numpy(),
        "msi_mod": msi_output.latent_mod.z.cpu().numpy(),
        "msi_s": msi_output.latent_s.z.cpu().numpy(),
    }


def train_mvae(model, mdata, params=TrainParams()):
    train_mdata, test_mdata = split_into_train_test(
        mdata,
        params.train_size,
        sample=params.leave_sample_out,
        batch_split=params.batch_split,
    )

    train_loader, train_loader_pairs = mudata_to_dataloader(
        train_mdata,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )
    test_loader, test_loader_pairs = mudata_to_dataloader(
        test_mdata,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )

    model.to(model.device)

    epoch_history = train(
        train_loader,
        train_loader_pairs,
        test_loader,
        test_loader_pairs,
        params,
    )
    model.eval()
    return model, epoch_history


def train(
    model: MVAE,
    train_loader: torch.utils.data.DataLoader,
    train_loader_pairs: torch.utils.data.DataLoader,
    test_loader=None,
    test_loader_pairs=None,
    params: TrainParams = TrainParams(),
):
    # Initialize Tensorboard summary writer
    writer = SummaryWriter(
        "logs/mvae" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    epoch_hist = {"train_loss": [], "valid_loss": []}
    optimizer = optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=5e-4
    )
    train_ES = EarlyStopping(patience=params.train_patience, verbose=True, mode="train")
    if test_loader:
        valid_ES = EarlyStopping(
            patience=params.test_patience, verbose=True, mode="valid"
        )

    # Train
    it = 0
    for epoch in range(params.n_epochs):
        torch.set_num_threads(16)
        model.train()
        for model_input, model_input_pairs in tqdm(
            zip(train_loader, train_loader_pairs), total=len(train_loader)
        ):
            optimizer.zero_grad()
            loss = Loss(params.beta, dropout=True)

            # Send input to device
            model_input: ModelInputT = {
                k: v.to(model.device) for k, v in model_input.items()
            }
            model_input_pairs: ModelInputT = {
                k: v.to(model.device) for k, v in model_input_pairs.items()
            }
            optimizer.zero_grad()

            model_output: ModelOutputT = model.forward(model_input)
            model_output_pairs: ModelOutputT = model.forward(model_input_pairs)

            loss.calculate_private(model_input, model_output)
            loss.calculate_shared(model_input_pairs, model_output_pairs)

            loss_values = loss.values

            loss.backward()
            optimizer.step()

            log_loss(writer, loss_values, it, train=True)
            it += 1

        # Get epoch loss
        epoch_loss = loss_values["loss"] / len(train_loader.dataset.indices)
        epoch_hist["train_loss"].append(epoch_loss)
        train_ES(epoch_loss)

        # Eval
        if test_loader:
            torch.save(model.state_dict(), "mvae_params.pt")

            test_dict = test_model(model, test_loader, test_loader_pairs)
            test_loss = test_dict["loss"]
            epoch_hist["valid_loss"].append(test_loss)
            valid_ES(test_loss)
            log_loss(writer, test_dict, epoch + 1, train=False)

    return epoch_hist


def test_model(model, loader, loader_pairs) -> Dict[str, float]:
    model.eval()
    latents = []
    mod_ids = []
    batch_rna_ids = []
    batch_msi_ids = []
    loss_values = []
    with torch.no_grad():
        for data, data_pairs in tqdm(zip(loader, loader_pairs), total=len(loader)):
            data = {k: v.to(model.device) for k, v in data.items()}
            data_pairs = {k: v.to(model.device) for k, v in data_pairs.items()}

            model_output: ModelOutputT = model.forward(data)
            latents.append(extract_latent(model_output))
            mod_ids.append(data["mod_id"].cpu().numpy())
            batch_rna_ids.append(data["batch_rna"].cpu().numpy())
            batch_msi_ids.append(data["batch_msi"].cpu().numpy())

            loss = Loss(beta=model.params.beta, dropout=model.params.dropout)

            model_output_pairs = model.forward(data_pairs)
            loss.calculate_private(data, model_output)
            loss.calculate_shared(data_pairs, model_output_pairs)
            loss_values.append(loss.values)

    last_loss = loss_values[-1]
    loss_private, loss_shared, loss_rna_batch, loss_msi_batch = (
        last_loss["private"],
        last_loss["shared"],
        last_loss["rna_batch"],
        last_loss["msi_batch"],
    )
    return {
        "loss": (loss_private + loss_shared) / len(loader),
        "loss_private": loss_private / len(loader),
        "loss_shared": loss_shared / len(loader),
        "loss_rna_batch": loss_rna_batch / len(loader),
        "loss_msi_batch": loss_msi_batch / len(loader),
    }


def predict(
    model: MVAE,
    adata,
    batch_size=1024,
    modality=None,
    indices=None,
    return_mean=False,
    use_gpu=True,
):
    """
    Project data into latent space. Inspired by SCVI.

    Parameters
    ----------
    adata
        scanpy single-cell dataset
    indices
        indices of the subset of cells to be encoded
    return_mean
        whether to use the mean of the multivariate gaussian or samples
    """
    model.to(model.device)
    train_loader, train_loader_pairs = _anndata_loader(
        adata,
        mod1_obsm=mod1_obsm,
        mod2_obsm=mod2_obsm,
        batch_size=self.batch_size,
        shuffle=False,
    )

    x1_poe = []
    x2_poe = []
    x1 = []
    x2 = []
    x1_2 = []
    x2_1 = []
    x1_batch_free = []
    x2_batch_free = []
    with torch.no_grad():
        self.eval()
        for tensors in tqdm(train_loader):
            tensors = {k: v.to(dev) for k, v in tensors.items()}
            model_output = self.forward(tensors)
            x1_poe += [model_output["x1_poe"].detach().cpu()]
            x2_poe += [model_output["x2_poe"].detach().cpu()]
            x1 += [model_output["x1"].detach().cpu()]
            x2 += [model_output["x2"].detach().cpu()]
            x1_2 += [model_output["x1_2"].detach().cpu()]
            x2_1 += [model_output["x2_1"].detach().cpu()]
            x1_batch_free += [model_output["x1_batch_free"].detach().cpu()]
            x2_batch_free += [model_output["x2_batch_free"].detach().cpu()]

    return x1_poe, x2_poe, x1, x2, x1_2, x2_1, x1_batch_free, x2_batch_free


@torch.no_grad()
def to_latent(
    self,
    adata,
    mod1_obsm=None,
    mod2_obsm=None,
    batch_size=1024,
    modality=None,
    indices=None,
    return_mean=False,
    use_gpu=True,
):
    """
    Project data into latent space. Inspired by SCVI.

    Parameters
    ----------
    adata
        scanpy single-cell dataset
    indices
        indices of the subset of cells to be encoded
    return_mean
        whether to use the mean of the multivariate gaussian or samples
    """
    dev = torch.device("cuda") if use_gpu else torch.device("cpu")
    self.to(dev)
    #        sc_dl, _ = _anndata_loader(adata, mod1_obsm=mod1_obsm, mod2_obsm=mod2_obsm, batch_size=batch_size, shuffle=False)
    # train_mdata, test_mdata = _anndata_splitter(adata, train_size=1)
    train_loader, train_loader_pairs = _anndata_loader(
        adata,
        mod1_obsm=mod1_obsm,
        mod2_obsm=mod2_obsm,
        batch_size=self.batch_size,
        shuffle=False,
    )
    #        test_loader, test_loader_pairs = _anndata_loader(test_mdata, mod1_obsm=mod1_obsm, mod2_obsm=mod2_obsm, batch_size=self.batch_size, shuffle=False)

    latent_z = []
    latent_z1_s = []
    latent_z2_s = []
    latent_z1_p = []
    latent_z2_p = []
    latent_z1_p_mod = []
    latent_z2_p_mod = []
    with torch.no_grad():
        self.eval()
        for tensors in tqdm(train_loader):
            tensors = {k: v.to(dev) for k, v in tensors.items()}
            model_output = self.forward(tensors)

            latent_z += [model_output["z"].cpu()]
            latent_z1_s += [model_output["z1_s"].cpu().numpy()]
            latent_z2_s += [model_output["z2_s"].cpu().numpy()]
            latent_z1_p += [model_output["z1_p"].cpu().numpy()]
            latent_z2_p += [model_output["z2_p"].cpu().numpy()]
            latent_z1_p_mod += [model_output["z1_p_mod"].cpu().numpy()]
            latent_z2_p_mod += [model_output["z2_p_mod"].cpu().numpy()]

    return (
        latent_z,
        latent_z1_s,
        latent_z2_s,
        latent_z1_p,
        latent_z1_p_mod,
        latent_z2_p,
        latent_z2_p_mod,
    )  # , train_mdata
