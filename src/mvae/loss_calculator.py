from typing import Callable, Dict

from src.loss import get_loss_fun, mmd, mse, compute_lisi

from src.harmony import harmonize

from src.latent import Latent

from src.mvae.types import (
    Modality,
    ModalityOutput,
    ModelInputT,
    ModelOutputT,
    ObsModalityMembership,
)
import torch
from torch.autograd import Variable


class LossCalculator:
    private = None
    shared = None
    batch_integration = None

    mmd = None
    rna = None
    msi = None
    rna_kl_p = None
    rna_kl_mod = None
    rna_kl_s = None
    msi_kl_p = None
    msi_kl_mod = None
    msi_kl_s = None

    loss_rna_msi = None
    loss_msi_rna = None

    recovered_rna_poe = None
    recovered_msi_poe = None

    kl = None

    beta: float
    loss_function: Callable = mse
    dropout: bool = False

    def __init__(self, beta, loss_function="mse", dropout=True):
        self.beta = beta
        self.loss_function = get_loss_fun(loss_function)
        self.dropout = dropout

    @property
    def total_loss(self) -> torch.Tensor:
        return self.private + self.shared + self.batch_integration

    @property
    def values(self) -> Dict[str, float]:
        return {
            "private": self.private.item(),
            "shared": self.shared.item(),
            "batch_integration": self.batch_integration.item(),
            "mmd": self.mmd.item(),
            "rna": self.rna.item(),
            "msi": self.msi.item(),
            "rna_kl_p": self.rna_kl_p.item(),
            "rna_kl_mod": self.rna_kl_mod.item(),
            "rna_kl_s": self.rna_kl_s.item(),
            "msi_kl_p": self.msi_kl_p.item(),
            "msi_kl_mod": self.msi_kl_mod.item(),
            "msi_kl_s": self.msi_kl_s.item(),
            "loss_rna_msi": self.loss_rna_msi.item(),
            "loss_msi_rna": self.loss_msi_rna.item(),
            "recovered_rna_poe": self.recovered_rna_poe.item(),
            "recovered_msi_poe": self.recovered_msi_poe.item(),
            "kl": self.kl.item(),
        }

    def calculate_private(
        self,
        model_input: ModelInputT,
        model_output: ModelOutputT,
    ) -> None:
        mod = model_input["mod_id"]
        rna_output = ModalityOutput.from_dict(model_output[Modality.rna.name])
        msi_output = ModalityOutput.from_dict(model_output[Modality.msi.name])
        rna_idxs = (mod == ObsModalityMembership.ONLY_MOD1) | (
            mod == ObsModalityMembership.PAIRED
        )
        msi_idxs = (mod == ObsModalityMembership.ONLY_MOD2) | (
            mod == ObsModalityMembership.PAIRED
        )
        (
            self.rna,
            self.rna_kl_p,
            self.rna_kl_mod,
            self.rna_kl_s,
        ) = self._loss_mod(model_input[Modality.rna.name], rna_output, rna_idxs)
        (
            self.msi,
            self.msi_kl_p,
            self.msi_kl_mod,
            self.msi_kl_s,
        ) = self._loss_mod(model_input[Modality.msi.name], msi_output, msi_idxs)

        self.private = (
            self.rna
            + self.msi
            + (self.rna_kl_p + self.rna_kl_s + self.msi_kl_p + self.msi_kl_s)
        )

    def _loss_mod(
        self,
        modality_input,
        modality_output: ModalityOutput,
        mod_idxs,
    ):
        """
        Calculates private loss components (MSE and KL) for one modality.
        """
        kld_p = modality_output.latent_p.kld()
        kld_mod = modality_output.latent_mod.kld()
        kld_s = modality_output.latent_s.kld()

        x_pred = modality_output.x[mod_idxs]
        x_real = torch.squeeze(modality_input[mod_idxs])

        loss = self.loss_function(x_pred, x_real, self.dropout)
        return (
            torch.mean(loss),
            self.beta * torch.mean(kld_p),
            self.beta * torch.mean(kld_mod),
            self.beta * torch.mean(kld_s),
        )

    def calculate_shared(
        self, model_input: ModelInputT, model_output: ModelOutputT
    ) -> None:
        """
        Initializes Maximum Mean Discrepancy(MMD) between model_input and output.
        - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.
        """
        rna_output = ModalityOutput.from_dict(model_output[Modality.rna.name])
        msi_output = ModalityOutput.from_dict(model_output[Modality.msi.name])
        mod = model_input["mod_id"]
        rna_idxs = (mod == ObsModalityMembership.ONLY_MOD1) | (
            mod == ObsModalityMembership.PAIRED
        )
        msi_idxs = (mod == ObsModalityMembership.ONLY_MOD2) | (
            mod == ObsModalityMembership.PAIRED
        )
        rna_real = torch.squeeze(model_input[Modality.rna.name])
        msi_real = torch.squeeze(model_input[Modality.msi.name])

        self.loss_rna_msi = torch.mean(
            self.loss_function(
                model_output["rna_msi_loss"][msi_idxs],
                msi_real[msi_idxs],
                dropout=self.dropout,
            )
        )
        self.loss_msi_rna = torch.mean(
            self.loss_function(
                model_output["msi_rna_loss"][rna_idxs],
                rna_real[rna_idxs],
                dropout=self.dropout,
            )
        )

        self.recovered_rna_poe = torch.mean(
            self.loss_function(
                model_output["rna_poe"][rna_idxs],
                rna_real[rna_idxs],
                dropout=self.dropout,
            )
        )
        self.recovered_msi_poe = torch.mean(
            self.loss_function(
                model_output["msi_poe"][msi_idxs],
                msi_real[msi_idxs],
                dropout=self.dropout,
            )
        )

        self.kl = self.beta * torch.mean(Latent(**model_output["poe_latent"]).kld())

        alphas = [
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            100,
            1e3,
            1e4,
            1e5,
            1e6,
        ]

        alphas = Variable(torch.FloatTensor(alphas)).to(device=torch.device("cuda"))

        self.mmd = mmd(rna_output.latent_s.z, msi_output.latent_s.z, alphas)

        self.shared = (
            self.loss_rna_msi
            + self.loss_msi_rna
            + self.recovered_rna_poe
            + self.recovered_msi_poe
            + self.kl
        )

    def calculate_batch_integration_loss(
        self,
        input: ModelInputT,
        output: ModelOutputT,
        perplexity: float = 30,
        device="cuda",
    ):
        """
        Tries to correct the POE latent space for batch effects with Harmony and calculates loss
        as LISI (Local Inverse Simpson Index) score.
        """
        batch_id = input["batch_id1"]
        poe = output["poe_latent"]["z"]

        # poe_corrected = harmonize(poe, batch_id, device_type=device)
        self.batch_integration = 1 / compute_lisi(poe, batch_id, perplexity)
