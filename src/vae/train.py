from typing import Dict, List, Optional, Tuple

from src.vae.types import VAEInputT, VAEOutputT

from src.vae.dataloader import adata_to_dataloader, setup_batch_key
from src.vae.loss_calculator import LossCalculator
import numpy as np
import torch
from tqdm import tqdm
from src.types import TrainParams
from src.utils import EarlyStopping, log_loss
from src.vae.model import VAE
from anndata import AnnData
from torch.utils.tensorboard import SummaryWriter
from torch import optim


def split_into_train_test(
    adata: AnnData, train_size: float = 0.8
) -> Tuple[AnnData, Optional[AnnData]]:
    assert 0 < train_size <= 1, "train_size must be between 0 and 1"

    idxs = np.arange(adata.shape[0])
    np.random.shuffle(idxs)

    n = int(train_size * len(idxs))
    train_idx = idxs[:n]
    test_idx = idxs[n:] if len(idxs) != n else []

    train_adata = adata.copy()[train_idx, :]
    test_adata = adata.copy()[test_idx, :] if len(test_idx) != 0 else None
    return train_adata, test_adata


def train_vae(
    model: VAE,
    adata: AnnData,
    batch_keys: List[str] = ["sample"],
    params: TrainParams = TrainParams(),
):
    batch_key_dict = setup_batch_key(adata, batch_keys)
    train_adata, test_adata = split_into_train_test(adata, train_size=params.train_size)
    train_loader = adata_to_dataloader(
        train_adata,
        batch_key_dict,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )
    if test_adata:
        test_loader = adata_to_dataloader(
            test_adata,
            batch_key_dict,
            batch_size=params.batch_size,
            shuffle=params.shuffle,
        )
    else:
        test_loader = None

    model.to(model.device)

    epoch_history = _train(
        model,
        train_loader,
        test_loader,
        batch_key_dict,
        params,
    )
    return epoch_history


def _train(
    model: VAE,
    train_loader,
    test_loader=None,
    batch_key_setup: Dict[str, Tuple[str, int]] = {},
    params: TrainParams = TrainParams(),
):
    # params file name will have time of training start
    params_file = params.get_params_file()

    writer = SummaryWriter("logs/" + params_file)

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
        dropout=params.dropout,
    )
    for epoch in range(params.n_epochs):
        torch.set_num_threads(16)
        model.train()
        epoch_loss = 0
        for model_input in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()

            # Send input to device
            model_input: VAEInputT = {
                k: v.to(model.device) for k, v in model_input.items()
            }

            model_output = model.forward(model_input)

            loss_calculator.calculate_private(model_input, model_output)

            if params.add_lisi_loss:
                loss_calculator.calculate_batch_integration_loss(
                    model_input, model_output
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
            log_loss(writer, test_loss, epoch + 1, train=False)

    writer.close()
    return epoch_hist


def test_model(
    model: VAE,
    loader,
    params: TrainParams,
    loss_calculator: LossCalculator,
) -> Dict[str, float]:
    model.eval()
    loss_val = 0
    i = 0
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            data = {k: v.to(model.device) for k, v in data.items()}
            model_output = model.forward(data)

            loss_calculator.calculate_private(data, model_output)
            if params.add_lisi_loss:
                loss_calculator.calculate_batch_integration_loss(data, model_output)

            loss_value = loss_calculator.total_loss.item()
            loss_val += loss_value
            i += 1

    return {
        "Test loss": loss_val / i,
        "batch_integration_loss": loss_calculator.values.get("batch_integration", 0),
    }


def predict(
    model: VAE,
    adata: AnnData,
    batch_keys: List[str] = ["sample"],
    params: TrainParams = TrainParams(),
):
    batch_key_setup = setup_batch_key(adata, batch_keys)
    model.to(model.device)
    train_loader = adata_to_dataloader(
        adata,
        batch_size=params.batch_size,
        batch_keys=batch_key_setup,
        shuffle=False,
    )
    y = []

    with torch.no_grad():
        model.eval()
        for tensors in tqdm(train_loader):
            tensors = {k: v.to(model.device) for k, v in tensors.items()}
            model_output: VAEOutputT = model.forward(tensors)
            y.append(model_output["x"].detach().cpu())

    return y


def to_latent(
    model: VAE,
    adata: AnnData,
    batch_keys: List[str] = ["sample"],
    params: TrainParams = TrainParams(),
):
    batch_key_setup = setup_batch_key(adata, batch_keys)
    model.to(model.device)
    train_loader = adata_to_dataloader(
        adata,
        batch_size=params.batch_size,
        batch_keys=batch_key_setup,
        shuffle=False,
    )

    latent = []

    with torch.no_grad():
        model.eval()
        for tensors in tqdm(train_loader):
            tensors = {k: v.to(model.device) for k, v in tensors.items()}
            model_output = model.forward(tensors)
            latent.append(model_output["latent"]["z"].cpu())

    return latent
