from dataclasses import dataclass
import datetime
from typing import Any, Dict, Optional, Tuple

from src.batch_correct import harmony_correct

from src.dataloader import mudata_to_dataloader

from src.utils import setup_mudata, split_into_train_test

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
    shuffle: bool = True
    batch_split: Any = None
    leave_sample_out: Optional[str] = None  # name of sample to leave out in training
    train_patience: int = 20
    test_patience: int = 20
    rna_loss: str = "mse"
    msi_loss: str = "mse"
    dropout: bool = True


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

    def __call__(self, val_loss: float, epoch: int):
        score = val_loss
        if self._has_score_improved(score):
            self.best_score = score
            self.counter = 0

            if self.verbose:
                print(
                    f"EarlyStopping (epoch: {epoch}): {self.mode} loss improved to {score}",
                    flush=True,
                )
            return

        self.counter += 1
        if self.counter >= self.patience:
            print(
                f"EarlyStopping (epoch: {epoch}): {self.mode} loss hasn't improved: {score}. Stopping.",
                flush=True,
            )
            self.early_stop = True

    def _has_score_improved(self, current_score) -> bool:
        return self.best_score is None or current_score < self.best_score - self.delta


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
        "poe": model_output["poe_latent"]["z"].cpu(),
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
        "rna_poe": model_output["rna_poe"].detach().cpu(),
        "msi_poe": model_output["msi_poe"].detach().cpu(),
        "rna": rna_output.x.detach().cpu(),
        "msi": msi_output.x.detach().cpu(),
        "trans_loss_rna_msi": model_output["rna_msi_loss"].detach().cpu(),
        "trans_loss_msi_rna": model_output["msi_rna_loss"].detach().cpu(),
        "rna_batch_free": model_output["rna_batch_free"].detach().cpu(),
        "msi_batch_free": model_output["msi_batch_free"].detach().cpu(),
    }


def train_mvae(
    model: MVAE, mdata: MuData, params=TrainParams()
) -> Tuple[MVAE, Dict[str, Any]]:
    setup_mudata(mdata)
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
        model,
        train_loader,
        train_loader_pairs,
        test_loader,
        test_loader_pairs,
        params,
    )
    torch.save(model.state_dict(), "mvae_params.pt")
    return epoch_history


def train(
    model: MVAE,
    train_loader: torch.utils.data.DataLoader,
    train_loader_pairs: torch.utils.data.DataLoader,
    test_loader=None,
    test_loader_pairs=None,
    params: TrainParams = TrainParams(),
):
    # Initialize Tensorboard summary writer
    writer = None
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
        epoch_loss = 0
        for model_input, model_input_pairs in tqdm(
            zip(train_loader, train_loader_pairs), total=len(train_loader)
        ):
            optimizer.zero_grad()
            loss = Loss(beta=model.params.beta, dropout=params.dropout)

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

            corr = harmony_correct(model_input_pairs, model_output_pairs, model.device)

            loss_values = loss.values
            epoch_loss += loss_values["loss"]

            loss.backward()
            optimizer.step()

            log_loss(writer, loss_values, it, train=True)
            it += 1

        # Get epoch loss
        epoch_loss = epoch_loss / len(train_loader.dataset.indices)
        epoch_hist["train_loss"].append(epoch_loss)

        train_ES(epoch_loss, epoch + 1)
        if not test_loader and train_ES.early_stop:
            # Use train early stop only if no test set is provided
            break

        # Eval
        if test_loader:
            # torch.save(model.state_dict(), "mvae_params.pt")
            test_loss = test_model(model, test_loader, test_loader_pairs, params)
            epoch_hist["valid_loss"].append(test_loss)
            valid_ES(test_loss, epoch + 1)
            if valid_ES.early_stop:
                break
            log_loss(writer, {"Test loss": test_loss}, epoch + 1, train=False)

    writer.close()
    return epoch_hist


def test_model(
    model: MVAE, loader, loader_pairs, params: TrainParams
) -> Dict[str, float]:
    model.eval()
    loss_val = 0
    i = 0
    with torch.no_grad():
        for data, data_pairs in tqdm(zip(loader, loader_pairs), total=len(loader)):
            data = {k: v.to(model.device) for k, v in data.items()}
            data_pairs = {k: v.to(model.device) for k, v in data_pairs.items()}

            model_output: ModelOutputT = model.forward(data)

            loss = Loss(beta=model.params.beta, dropout=params.dropout)

            model_output_pairs = model.forward(data_pairs)
            loss.calculate_private(data, model_output)
            loss.calculate_shared(data_pairs, model_output_pairs)
            loss_val += loss.values["loss"]
            i += 1

    return loss_val / i


def predict(
    model: MVAE,
    mudata: MuData,
    params: TrainParams,
):
    model.to(model.device)
    train_loader, _ = mudata_to_dataloader(
        mudata,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )
    rna_poe = []
    msi_poe = []
    rna = []
    msi = []
    trans_loss_rna_msi = []
    trans_loss_msi_rna = []
    rna_batch_free = []
    msi_batch_free = []

    with torch.no_grad():
        model.eval()
        for tensors in tqdm(train_loader):
            tensors = {k: v.to(model.device) for k, v in tensors.items()}
            model_output: ModelOutputT = model.forward(tensors)
            y = extract_y(model_output)
            rna_poe.append(y["rna_poe"])
            msi_poe.append(y["msi_poe"])
            rna.append(y["rna"])
            msi.append(y["msi"])
            trans_loss_rna_msi.append(y["trans_loss_rna_msi"])
            trans_loss_msi_rna.append(y["trans_loss_msi_rna"])
            rna_batch_free.append(y["rna_batch_free"])
            msi_batch_free.append(y["msi_batch_free"])
    return (
        rna_poe,
        msi_poe,
        rna,
        msi,
        trans_loss_rna_msi,
        trans_loss_msi_rna,
        rna_batch_free,
        msi_batch_free,
    )


def to_latent(
    model: MVAE,
    mudata: MuData,
    params: TrainParams,
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

    poe = []
    rna_p = []
    msi_p = []
    rna_mod = []
    msi_mod = []
    rna_s = []
    msi_s = []

    with torch.no_grad():
        model.eval()
        for tensors in tqdm(train_loader):
            tensors = {k: v.to(model.device) for k, v in tensors.items()}
            model_output = model.forward(tensors)
            latent = extract_latent(model_output)
            poe.append(latent["poe"])
            rna_p.append(latent["rna_p"])
            msi_p.append(latent["msi_p"])
            rna_mod.append(latent["rna_mod"])
            msi_mod.append(latent["msi_mod"])
            rna_s.append(latent["rna_s"])
            msi_s.append(latent["msi_s"])

    return poe, rna_p, msi_p, rna_mod, msi_mod, rna_s, msi_s
