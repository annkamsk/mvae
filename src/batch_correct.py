import numpy as np
from src.types import ModelInputT, ModelOutputT
from src.harmony import (
    initialize_centroids,
    clustering,
    correction,
    is_convergent_harmony,
)
from src.constants import BATCH_KEY
from src.utils import one_hot_tensor

from torch.nn.functional import normalize
import torch


def harmony_correct(
    model_input: ModelInputT, model_output: ModelOutputT, device: str
) -> np.array:
    poe = model_output["poe_latent"]["z"].detach().cpu().numpy()

    batch_id = model_input["batch_id1"]
    return harmonize(poe, batch_id, device_type=device)


def harmonize(
    X: np.array,
    batch_ids: torch.Tensor,
    n_clusters: int = None,
    max_iter_harmony: int = 10,
    max_iter_clustering: int = 200,
    tol_harmony: float = 1e-4,
    tol_clustering: float = 1e-5,
    ridge_lambda: float = 1.0,
    sigma: float = 0.1,
    block_proportion: float = 0.05,
    theta: float = 2.0,
    tau: int = 0,
    correction_method: str = "fast",
    random_state: int = 0,
    device_type: str = "cpu",
    n_jobs: int = -1,
    verbose: bool = True,
) -> np.array:
    """
    Integrate data using Harmony algorithm.
    Adapted from: https://github.com/lilab-bcb/harmony-pytorch to work with batch ids encoded in Tensor.

    Parameters
    ----------

    X: ``numpy.array``
        The input embedding with rows for cells (N) and columns for embedding coordinates (d).

    batch_ids: ``torch.Tensor``
        Tensor of N length with batch identification for the cells.

    n_clusters: ``int``, optional, default: ``None``
        Number of clusters used in Harmony algorithm. If ``None``, choose the minimum of 100 and N / 30.

    max_iter_harmony: ``int``, optional, default: ``10``
        Maximum iterations on running Harmony if not converged.

    max_iter_clustering: ``int``, optional, default: ``200``
        Within each Harmony iteration, maximum iterations on the clustering step if not converged.

    tol_harmony: ``float``, optional, default: ``1e-4``
        Tolerance on justifying convergence of Harmony over objective function values.

    tol_clustering: ``float``, optional, default: ``1e-5``
        Tolerance on justifying convergence of the clustering step over objective function values within each Harmony iteration.

    ridge_lambda: ``float``, optional, default: ``1.0``
        Hyperparameter of ridge regression on the correction step.

    sigma: ``float``, optional, default: ``0.1``
        Weight of the entropy term in objective function.

    block_proportion: ``float``, optional, default: ``0.05``
        Proportion of block size in one update operation of clustering step.

    theta: ``float``, optional, default: ``2.0``
        Weight of the diversity penalty term in objective function.

    tau: ``int``, optional, default: ``0``
        Discounting factor on ``theta``. By default, there is no discounting.

    correction_method: ``string``, optional, default: ``fast``
        Choose which method for the correction step: ``original`` for original method, ``fast`` for improved method. By default, use improved method.

    random_state: ``int``, optional, default: ``0``
        Random seed for reproducing results.

    device_type: ``str``, optional, default: ``cpu``
        If ``cuda``, use GPU.

    n_jobs: ``int``, optional, default ``-1``
        How many CPU threads to use. By default, use all physical cores. If 'use_gpu' is True, this option only affects the KMeans step.

    verbose: ``bool``, optional, default ``True``
        If ``True``, print verbose output.

    Returns
    -------
    ``numpy.array``
        The integrated embedding by Harmony, of the same shape as the input embedding.
    """
    assert isinstance(X, np.ndarray)

    Z = torch.tensor(X, dtype=torch.float, device=device_type)
    Z_norm = normalize(Z, p=2, dim=1)
    n_cells = Z.shape[0]

    _, N_b = torch.unique(batch_ids, sorted=True, return_counts=True)
    Pr_b = N_b.view(-1, 1) / n_cells

    n_batches = N_b.size(dim=0)

    Phi = one_hot_tensor(batch_ids, n_batches, device_type)

    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))

    theta = torch.tensor([theta], dtype=torch.float, device=device_type).expand(
        n_batches
    )

    if tau > 0:
        theta = theta * (1 - torch.exp(-N_b / (n_clusters * tau)) ** 2)

    theta = theta.view(1, -1)

    assert block_proportion > 0 and block_proportion <= 1
    assert correction_method in ["fast", "original"]

    np.random.seed(random_state)

    # Initialize centroids
    R, E, O, objectives_harmony = initialize_centroids(
        Z_norm,
        n_clusters,
        sigma,
        Pr_b,
        Phi,
        theta,
        None,
        device_type,
        n_jobs,
    )

    if verbose:
        print("\tInitialization is completed.")

    for i in range(max_iter_harmony):
        clustering(
            Z_norm,
            Pr_b,
            Phi,
            R,
            E,
            O,
            n_clusters,
            theta,
            tol_clustering,
            objectives_harmony,
            max_iter_clustering,
            sigma,
            block_proportion,
            device_type,
        )
        Z_hat = correction(Z, R, Phi, O, ridge_lambda, correction_method, device_type)
        Z_norm = normalize(Z_hat, p=2, dim=1)

        if verbose:
            print(
                "\tCompleted {cur_iter} / {total_iter} iteration(s).".format(
                    cur_iter=i + 1,
                    total_iter=max_iter_harmony,
                )
            )

        if is_convergent_harmony(objectives_harmony, tol=tol_harmony):
            if verbose:
                print("Reach convergence after {} iteration(s).".format(i + 1))
            break

    if device_type == "cpu":
        return Z_hat.numpy()
    else:
        return Z_hat.cpu().numpy()
