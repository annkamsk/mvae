from dataclasses import dataclass
import datetime

import numpy as np
from tqdm import tqdm
from src.model import MVAE
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn, optim


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


@dataclass
class Loss:
    loss_value = 0
    loss_rec_rna_v = 0
    loss_rec_msi_v = 0
    loss_trls_msi_v = 0
    loss_trls_rna_v = 0
    loss_rec_rna_poe_v = 0
    loss_rec_msi_poe_v = 0
    loss_kl_v = 0
    loss_kl1_p_v = 0
    loss_kl1_p_mod_v = 0
    loss_kl1_s_v = 0
    loss_kl2_p_v = 0
    loss_kl2_p_mod_v = 0
    loss_kl2_s_v = 0
    loss_mmd_v = 0
    loss_shared_v = 0
    loss_private_v = 0
    loss_private_mod_v = 0
    loss_batch_mod1_v = 0
    loss_batch_mod2_v = 0
    loss_cos1_v = 0
    loss_cos2_v = 0

    ####################################
    loss_batch_1_v = 0
    loss_batch_2_v = 0
    ####################################


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
            self.early_stop = True

    def _has_score_improved(self, current_score) -> bool:
        return self.best_score is None or current_score > self.best_score - self.delta


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
            loss = Loss()

            # Send input to device
            model_input = {k: v.to(model.device) for k, v in model_input.items()}
            model_input_pairs = {
                k: v.to(model.device) for k, v in model_input_pairs.items()
            }
            optimizer.zero_grad()

            model_output = self.forward(model_input)
            model_output_pairs = self.forward(model_input_pairs)

            (
                loss_private,
                loss_rec_rna,
                loss_rec_msi,
                loss_kl1_p,
                loss_kl1_s,
                loss_kl2_p,
                loss_kl2_s,
                loss_kl1_p_mod,
                loss_kl2_p_mod,
                loss_batch1,
                loss_batch2,
            ) = self.private_loss(model_input, model_output)
            (
                loss_shared,
                loss_mmd,
                loss_trls_msi,
                loss_trls_rna,
                loss_rec_rna_poe,
                loss_rec_msi_poe,
                loss_kl,
                loss_cos1,
                loss_cos2,
            ) = self.shared_loss(model_input_pairs, model_output_pairs)

            loss = loss_private + loss_shared  # + loss_batch_mod1 + loss_batch_mod2
            loss_value += loss.item()
            loss_rec_rna_v += loss_rec_rna.item()
            loss_rec_msi_v += loss_rec_msi.item()
            loss_trls_msi_v += loss_trls_msi.item()
            loss_trls_rna_v += loss_trls_rna.item()
            loss_rec_rna_poe_v += loss_rec_rna_poe.item()
            loss_rec_msi_poe_v += loss_rec_msi_poe.item()
            loss_kl_v += loss_kl.item()
            loss_kl1_p_v += loss_kl1_p.item()
            loss_kl1_p_mod_v += loss_kl1_p_mod.item()
            loss_kl1_s_v += loss_kl1_s.item()
            loss_kl2_p_v += loss_kl2_p.item()
            loss_kl2_p_mod_v += loss_kl2_p_mod.item()
            loss_kl2_s_v += loss_kl2_s.item()
            loss_shared_v += loss_shared.item()
            loss_private_v += loss_private.item()
            loss_mmd_v += loss_mmd.item()
            loss_cos1_v += loss_cos1
            loss_cos2_v += loss_cos2

            ####################################
            loss_batch_1_v += loss_batch1.item()
            loss_batch_2_v += loss_batch2.item()
            ####################################

            loss.backward()
            optimizer.step()

            writer.add_scalar("PoE_training/Loss", loss_value, it)
            writer.add_scalar("PoE_training/mse_rna", loss_rec_rna_v, it)
            writer.add_scalar("PoE_training/mse_msi", loss_rec_msi_v, it)
            writer.add_scalar("PoE_training/mse_rna_poe", loss_rec_rna_poe_v, it)
            writer.add_scalar("PoE_training/mse_msi_poe", loss_rec_msi_poe_v, it)
            writer.add_scalar("PoE_training/kl_loss", loss_kl_v, it)
            writer.add_scalar("PoE_training/kl1_p_loss", loss_kl1_p_v, it)
            writer.add_scalar("PoE_training/kl1_s_loss", loss_kl1_s_v, it)
            writer.add_scalar("PoE_training/kl1_p_mod_loss", loss_kl1_p_mod_v, it)
            writer.add_scalar("PoE_training/kl2_p_loss", loss_kl2_p_v, it)
            writer.add_scalar("PoE_training/kl2_s_loss", loss_kl2_s_v, it)
            writer.add_scalar("PoE_training/kl2_p_mod_loss", loss_kl2_p_mod_v, it)
            writer.add_scalar("PoE_training/mmd_loss", loss_mmd_v, it)
            writer.add_scalar("PoE_training/shared_loss", loss_shared_v, it)
            writer.add_scalar("PoE_training/private_loss", loss_private_v, it)
            writer.add_scalar("PoE_training/trls_rna_loss", loss_trls_rna_v, it)
            writer.add_scalar("PoE_training/trls_msi_loss", loss_trls_msi_v, it)
            writer.add_scalar("PoE_training/cos1_loss", loss_cos1_v, it)
            writer.add_scalar("PoE_training/cos2_loss", loss_cos2_v, it)
            ##############################################################################
            writer.add_scalar("PoE_training/loss_batch1", loss_batch_1_v, it)
            writer.add_scalar("PoE_training/loss_batch2", loss_batch_2_v, it)
            ##############################################################################

            it += 1

        # Get epoch loss
        epoch_loss = loss_value / len(train_loader.dataset.indices)
        epoch_hist["train_loss"].append(epoch_loss)
        train_ES(epoch_loss)
        # Eval
        if test_loader:
            self.eval()
            torch.save(self.state_dict(), "mvae_params.pt")
            test_dict = self._test_model(test_loader, test_loader_pairs, device)
            test_loss = test_dict["loss"]
            epoch_hist["valid_loss"].append(test_loss)
            valid_ES(test_loss)
            writer.add_scalar("PoE_training/test_loss", test_loss, epoch + 1)
            writer.add_scalar(
                "PoE_training/test_loss_shared", test_dict["loss_shared"], epoch + 1
            )
            writer.add_scalar(
                "PoE_training/test_loss_batch_mod1",
                test_dict["loss_batch_mod1"],
                epoch + 1,
            )
            writer.add_scalar(
                "PoE_training/test_loss_batch_mod2",
                test_dict["loss_batch_mod2"],
                epoch + 1,
            )

            print(
                "[Epoch %d] | loss: %.3f | loss_rna: %.3f |loss_msi: %.3f | test_loss: %.3f |"
                % (
                    epoch + 1,
                    epoch_loss,
                    loss_rec_rna_v / len(train_loader.dataset.indices),
                    loss_rec_msi_v / len(train_loader.dataset.indices),
                    test_loss,
                ),
                flush=True,
            )
            if valid_ES.early_stop or train_ES.early_stop:
                # print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                # break
                print("", end="")
            else:
                print(
                    "[Epoch %d] | loss: %.3f |" % (epoch + 1, epoch_loss),
                    flush=True,
                )
                # if train_ES.early_stop:
                # print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                # break
    return epoch_hist
