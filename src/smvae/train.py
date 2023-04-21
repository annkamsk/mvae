from typing import List

from OtF-prostate.src.smvae.types import ModelOutputT
from OtF-prostate.src.smvae.types import ModelInputT

from tqdm import tqdm
from OtF-prostate.src.utils import EarlyStopping

import torch
from src.smvae.dataloader import (
    mudata_to_dataloader,
    setup_mudata,
    split_into_train_test,
)
from src.types import TrainParams
from src.smvae.model import SMVAE
from mudata import MuData


def train_mvae(
    model: SMVAE,
    mdata: MuData,
    batch_keys: List[str] = ["sample"],
    params=TrainParams(),
) -> Tuple[MVAE, Dict[str, Any]]:
    setup_mudata(mdata, batch_keys)
    train_mdata, test_mdata = split_into_train_test(
        mdata,
        params.train_size,
        sample=params.leave_sample_out,
        batch_split=params.batch_split,
    )

    train_loader, train_loader_pairs = mudata_to_dataloader(
        train_mdata,
        batch_keys,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )
    test_loader, test_loader_pairs = mudata_to_dataloader(
        test_mdata,
        batch_keys,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
    )
    model.to(model.device)

    epoch_history = _train(
        model,
        train_loader,
        train_loader_pairs,
        test_loader,
        test_loader_pairs,
        params,
    )
    return epoch_history


def _train(
    model: SMVAE,
    train_loader: torch.utils.data.DataLoader,
    train_loader_pairs: torch.utils.data.DataLoader,
    test_loader=None,
    test_loader_pairs=None,
    batch_key_dict = {},
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
    loss_calculator = VAE_LossCalculator(
        beta=model.params.beta,
        dropout=params.dropout,
        batch_key_dict=batch_key_setup,
        summary_writer=writer,
    )
    for epoch in range(params.n_epochs):
        torch.set_num_threads(16)
        model.train()
        epoch_loss = 0
        for model_input, model_input_pairs in tqdm(
            zip(train_loader, train_loader_pairs), total=len(train_loader)
        ):
            optimizer.zero_grad()

            # Send input to device
            model_input: ModelInputT = {
                k: v.to(model.device) for k, v in model_input.items()
            }
            model_input_pairs: ModelInputT = {
                k: v.to(model.device) for k, v in model_input_pairs.items()
            }

            model_output: ModelOutputT = model.forward(model_input)
            model_output_pairs: ModelOutputT = model.forward(model_input_pairs)

            loss_calculator.calculate_private(model_input, model_output)
            loss_calculator.calculate_shared(model_input_pairs, model_output_pairs)
            if params.add_lisi_loss:
                loss_calculator.calculate_batch_integration_loss(
                    model_input, model_output, model_output_pairs
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

            test_loss = test_model(model, test_loader, test_loader_pairs, params, loss_calculator)
            epoch_hist["valid_loss"].append(test_loss["Test loss"])
            valid_ES(test_loss["Test loss"], epoch + 1)

            torch.save(model.state_dict(), params_file)


            if valid_ES.early_stop:
                break

            log_loss(writer, {"Test loss": test_loss}, epoch + 1, train=False)

    writer.close()
    return epoch_hist


def test_model(
    model: SMVAE, loader, loader_pairs, params: TrainParams, loss_calculator
) -> Dict[str, float]:
    model.eval()
    loss_val = 0
    i = 0
    with torch.no_grad():
        for data, data_pairs in tqdm(zip(loader, loader_pairs), total=len(loader)):
            data = {k: v.to(model.device) for k, v in data.items()}
            data_pairs = {k: v.to(model.device) for k, v in data_pairs.items()}

            model_output: ModelOutputT = model.forward(data)

            model_output_pairs = model.forward(data_pairs)
            loss_calculator.calculate_private(data, model_output)
            loss_calculator.calculate_shared(data_pairs, model_output_pairs)
            loss_calculator.calculate_batch_integration_loss(
                data_pairs, model_output_pairs, device=model.device
            )

            loss_value = loss_calculator.total_loss.item()
            loss_val += loss_value
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
