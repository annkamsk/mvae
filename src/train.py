from dataclasses import dataclass
import datetime
from typing import Dict, Optional

from src.dataloader import mudata_to_dataloader

from src.utils import split_into_train_test

from src.loss import Loss
from mudata import MuData
from src.types import ModalityOutput, ModelInputT, ModelOutputT

import numpy as np
from tqdm import tqdm
from src.model import MVAE
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim


@dataclass
class TrainParams:
    learning_rate: float = 1e-4
    n_epochs: int = 500
    train_size: float = 1.0  # proportion of training data, between 0 and 1
    batch_size: int = 128
    batch_split = None
    shuffle = True
    leave_sample_out: Optional[str] = None  # name of sample to leave out in training
    train_patience = 20
    test_patience = 20
    rna_loss = "mse"
    msi_loss = "bce"
    dropout = True


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


def extract_y(model_output: ModelOutputT):
    rna_output = ModalityOutput.from_dict(model_output["rna"])
    msi_output = ModalityOutput.from_dict(model_output["msi"])
    return {
        "rna_poe": rna_output.y_poe.detach().cpu(),
        "msi_poe": msi_output.y_poe.detach().cpu(),
        "rna": rna_output.y.detach().cpu(),
        "msi": msi_output.y.detach().cpu(),
        "trans_loss_rna_msi": model_output["rna_msi_loss"].detach().cpu(),
        "trans_loss_msi_rna": model_output["msi_rna_loss"].detach().cpu(),
        "rna_batch_free": rna_output.y_batch_free.detach().cpu(),
        "msi_batch_free": msi_output.y_batch_free.detach().cpu(),
    }


def train_mvae(model: MVAE, mdata: MuData, params=TrainParams()):
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
            loss = Loss(model.params.beta, params.dropout)

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
            # torch.save(model.state_dict(), "mvae_params.pt")
            test_dict = test_model(model, test_loader, test_loader_pairs)
            test_loss = test_dict["loss"]
            epoch_hist["valid_loss"].append(test_loss)
            valid_ES(test_loss)
            log_loss(writer, test_dict, epoch + 1, train=False)

    return epoch_hist


def test_model(
    model: MVAE, loader, loader_pairs, params: TrainParams
) -> Dict[str, float]:
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

            loss = Loss(beta=model.params.beta, dropout=params.dropout)

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
    mudata: MuData,
    params=TrainParams(batch_size=1024),
):
    model.to(model.device)
    train_loader, _ = mudata_to_dataloader(
        mudata,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )
    y = []
    with torch.no_grad():
        model.eval()
        for tensors in tqdm(train_loader):
            tensors = {k: v.to(model.device) for k, v in tensors.items()}
            model_output: ModelOutputT = model.forward(tensors)
            y.append(extract_y(model_output))
    return y


def to_latent(
    model: MVAE,
    mudata: MuData,
    params=TrainParams(batch_size=1024),
):
    """
    Projects data into latent space. Inspired by SCVI.
    """
    model.to(model.device)
    train_loader, _ = mudata_to_dataloader(
        mudata,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )

    latent = []
    with torch.no_grad():
        model.eval()
        for tensors in tqdm(train_loader):
            tensors = {k: v.to(model.device) for k, v in tensors.items()}
            model_output = model.forward(tensors)
            latent.append(extract_latent(model_output))

    return latent
