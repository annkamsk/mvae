from typing import Any, Dict, List

from src.smvae.types import ModalityOutput, ModelOutputT
from src.smvae.types import ModelInputT

from tqdm import tqdm
from src.utils import EarlyStopping, log_loss
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch
from src.smvae.dataloader import (
    mudata_to_dataloader,
    setup_mudata,
    split_into_train_test,
)
from src.types import Modality, TrainParams
from src.smvae.model import SMVAE
from src.smvae.loss_calculator import LossCalculator
from mudata import MuData


def train_smvae(
    model: SMVAE,
    mdata: MuData,
    batch_key: str = "sample",
    params=TrainParams(),
) -> Dict[str, Any]:
    batch_num = setup_mudata(mdata, batch_key)
    train_mdata, test_mdata = split_into_train_test(
        mdata,
        params.train_size,
    )

    train_loader = mudata_to_dataloader(
        train_mdata,
        batch_num,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )
    test_loader = mudata_to_dataloader(
        test_mdata,
        batch_num,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )
    model.to(model.device)

    epoch_history = _train(
        model,
        train_loader,
        test_loader,
        batch_num,
        params,
    )
    return epoch_history


def _train(
    model: SMVAE,
    train_loader: torch.utils.data.DataLoader,
    test_loader=None,
    batch_num=0,
    params: TrainParams = TrainParams(),
):
    # Initialize Tensorboard summary writer
    # params file name will have time of training start
    params_file = params.get_params_file()

    writer = SummaryWriter("logs/smvae" + params_file)

    epoch_hist = {"train_loss": [], "valid_loss": []}
    optimizer = optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=5e-4
    )
    train_ES = EarlyStopping(patience=params.train_patience, verbose=True, mode="train")
    if test_loader:
        valid_ES = EarlyStopping(
            patience=params.test_patience, verbose=True, mode="valid"
        )

    it = 0
    loss_calculator = LossCalculator(
        beta=model.params.beta,
        gamma=model.params.gamma,
        delta=model.params.delta,
        dropout=params.dropout,
        batch_num=batch_num,
        summary_writer=writer,
    )
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
            test_loss = test_model(model, test_loader, params, loss_calculator)
            epoch_hist["valid_loss"].append(test_loss["Test loss"])
            valid_ES(test_loss["Test loss"], epoch + 1)

            torch.save(model.state_dict(), params_file)

            if valid_ES.early_stop:
                break

            log_loss(
                writer,
                test_loss,
                epoch + 1,
                train=False,
            )

    writer.close()
    return epoch_hist


def test_model(
    model: SMVAE, loader, params: TrainParams, loss_calculator: LossCalculator
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


def extract_latent(model_output: ModelOutputT):
    rna_output = ModalityOutput.from_dict(model_output[Modality.rna.name])
    msi_output = ModalityOutput.from_dict(model_output[Modality.msi.name])
    return {
        "poe": model_output["poe_latent"]["z"].cpu(),
        "rna_p": rna_output.latent_p.z.cpu().numpy(),
        "rna_s": rna_output.latent_s.z.cpu().numpy(),
        "msi_p": msi_output.latent_p.z.cpu().numpy(),
        "msi_s": msi_output.latent_s.z.cpu().numpy(),
    }


def extract_y(model_output: ModelOutputT):
    mod1_output = ModalityOutput.from_dict(model_output[Modality.rna.name])
    mod2_output = ModalityOutput.from_dict(model_output[Modality.msi.name])
    return {
        "rna_poe": model_output["rna_poe"].detach().cpu(),
        "msi_poe": model_output["msi_poe"].detach().cpu(),
        "rna": mod1_output.x.detach().cpu(),
        "msi": mod2_output.x.detach().cpu(),
        "rna_msi_loss": model_output["rna_msi_loss"].detach().cpu(),
        "msi_rna_loss": model_output["msi_rna_loss"].detach().cpu(),
    }


def predict(
    model: SMVAE,
    mudata: MuData,
    batch_key: str,
    params: TrainParams,
):
    batch_num = setup_mudata(mudata, batch_key)
    model.to(model.device)
    train_loader, _ = mudata_to_dataloader(
        mudata,
        batch_num,
        batch_size=params.batch_size,
        shuffle=False,
    )
    mod1_poe = []
    mod2_poe = []
    mod1 = []
    mod2 = []
    mod1_mod2_loss = []
    mod2_mod1_loss = []

    with torch.no_grad():
        model.eval()
        for tensors in tqdm(train_loader):
            tensors = {k: v.to(model.device) for k, v in tensors.items()}
            model_output: ModelOutputT = model.forward(tensors)
            y = extract_y(model_output)
            mod1_poe.append(y["mod1_poe"])
            mod2_poe.append(y["mod2_poe"])
            mod1.append(y["mod1"])
            mod2.append(y["mod2"])
            mod1_mod2_loss.append(y["mod1_mod2_loss"])
            mod2_mod1_loss.append(y["mod2_mod1_loss"])

    return (
        mod1_poe,
        mod2_poe,
        mod1,
        mod2,
        mod1_mod2_loss,
        mod2_mod1_loss,
    )


def to_latent(
    model: SMVAE,
    mudata: MuData,
    batch_key: str,
    params: TrainParams,
):
    """
    Projects data into latent space. Inspired by SCVI.
    """
    batch_num = setup_mudata(mudata, batch_key)
    model.to(model.device)
    train_loader = mudata_to_dataloader(
        mudata,
        batch_num,
        batch_size=params.batch_size,
        shuffle=False,
    )

    poe = []
    rna_p = []
    msi_p = []
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
            rna_s.append(latent["rna_s"])
            msi_s.append(latent["msi_s"])

    return poe, rna_p, msi_p, rna_s, msi_s
