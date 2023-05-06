import datetime
from typing import Any, Dict, Tuple

from src.utils import EarlyStopping, log_loss
from src.types import TrainParams

from src.mvae.dataloader import (
    mudata_to_dataloader,
    setup_mudata,
    split_into_train_test,
)

from src.mvae.loss_calculator import LossCalculator as MVAE_LossCalculator
from mudata import MuData
from src.mvae.types import ModalityOutput, ModelInputT, ModelOutputT

from tqdm import tqdm
from src.mvae.model import MVAE
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim


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
        "rna_batch_free": model_output["rna_batch_free"].detach().cpu(),
        "msi_batch_free": model_output["msi_batch_free"].detach().cpu(),
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
    model: MVAE,
    mdata: MuData,
    params: TrainParams,
    batch_key: str = "sample",
) -> Dict[str, Any]:
    batch_num = setup_mudata(mdata, batch_key)
    train_mdata, test_mdata = split_into_train_test(
        mdata,
        params.train_size,
    )
    train_loader = mudata_to_dataloader(
        train_mdata,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )
    test_loader = mudata_to_dataloader(
        test_mdata,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )
    model.to(model.device)

    epoch_history = _train(
        model,
        params=params,
        train_loader=train_loader,
        test_loader=test_loader,
        batch_num=batch_num,
    )
    return epoch_history


def _train(
    model: MVAE,
    params: TrainParams,
    train_loader: torch.utils.data.DataLoader,
    test_loader=None,
    batch_num=0,
):
    # Initialize Tensorboard summary writer
    # params file name will have time of training start
    params_file = params.get_params_file()

    writer = SummaryWriter("logs/mvae" + params_file)

    epoch_hist = {"train_loss": [], "valid_loss": []}
    optimizer = optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=5e-4
    )
    train_ES = EarlyStopping(patience=params.train_patience, verbose=True, mode="train")
    if test_loader:
        valid_ES = EarlyStopping(
            patience=params.test_patience, verbose=True, mode="valid"
        )

    loss_calculator = MVAE_LossCalculator(
        beta=model.params.beta,
        gamma=model.params.gamma,
        delta=model.params.delta,
        dropout=params.dropout,
        batch_num=batch_num,
    )

    it = 0
    for epoch in range(params.n_epochs):
        torch.set_num_threads(16)
        model.train()
        epoch_loss = 0
        for model_input in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()

            # Send input to device
            model_input: ModelInputT = {
                k: v.to(model.device) for k, v in model_input.items()
            }

            model_output: ModelOutputT = model.forward(model_input)

            loss_calculator.calculate_private(model_input, model_output)
            loss_calculator.calculate_shared(model_input, model_output)
            if params.add_lisi_loss:
                loss_calculator.calculate_batch_integration_loss(
                    model_input,
                    model_output,
                    on_privates=params.inverse_lisi_on_private,
                )

            loss = loss_calculator.total_loss
            loss_value = loss.item()
            epoch_loss += loss_value

            loss.backward()
            optimizer.step()

            log_loss(
                writer,
                {"total_loss": loss_value, **loss_calculator.values},
                it,
                train=True,
            )
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
            torch.save(
                model.state_dict(),
                params_file,
            )
            test_loss = test_model(model, test_loader, params, loss_calculator)
            epoch_hist["valid_loss"].append(test_loss["Test loss"])
            valid_ES(test_loss["Test loss"], epoch + 1)
            if valid_ES.early_stop:
                break
            log_loss(writer, test_loss, epoch + 1, train=False)

    writer.close()
    return epoch_hist


def test_model(
    model: MVAE,
    loader,
    params: TrainParams,
    loss_calculator: MVAE_LossCalculator,
) -> Dict[str, float]:
    model.eval()
    loss_val = 0
    i = 0
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            data = {k: v.to(model.device) for k, v in data.items()}

            model_output: ModelOutputT = model.forward(data)

            loss_calculator.calculate_private(data, model_output)
            loss_calculator.calculate_shared(data, model_output)
            if params.add_lisi_loss:
                loss_calculator.calculate_batch_integration_loss(
                    data, model_output, on_privates=params.inverse_lisi_on_private
                )

            loss_value = loss_calculator.total_loss.item()
            loss_val += loss_value
            i += 1

    return {
        "Test loss": loss_val / i,
        "batch_integration_loss": loss_calculator.values.get("batch_integration", 0),
    }


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
    train_loader = mudata_to_dataloader(
        mudata,
        batch_size=params.batch_size,
        shuffle=False,
    )

    poe = []
    rna_p = []
    msi_p = []
    rna_mod = []
    msi_mod = []
    rna_s = []
    msi_s = []
    rna_batch_free = []
    msi_batch_free = []

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
            rna_batch_free.append(latent["rna_batch_free"])
            msi_batch_free.append(latent["msi_batch_free"])

    return (
        poe,
        rna_p,
        msi_p,
        rna_mod,
        msi_mod,
        rna_s,
        msi_s,
        rna_batch_free,
        msi_batch_free,
    )
