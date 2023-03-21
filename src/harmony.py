"""
Methods defined in this file are adapted from the following repository:
https://github.com/lilab-bcb/harmony-pytorch
"""

from typing import Tuple
import torch

import numpy as np
from torch.nn.functional import normalize, one_hot


def harmonize(
    X: torch.Tensor,
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
    correction_method: str = "original",
    device_type: str = "cuda",
    verbose: bool = False,
) -> torch.Tensor:
    """
    Parameters
    ----------

    X: ``torch.Tensor``
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

    Returns
    -------
    ``torch.Tensor``
        The integrated embedding by Harmony, of the same shape as the input embedding.
    """
    Z = X.to(dtype=torch.float)

    # L2 normalization
    Z_norm = normalize(Z, p=2, dim=1)
    n_cells = Z.shape[0]

    # number of items in each batch
    _, N_b = torch.unique(batch_ids, sorted=True, return_counts=True)
    # proportion of items in each batch
    Pr_b = N_b.view(-1, 1) / n_cells

    n_batches = N_b.size(dim=0)

    Phi = one_hot(batch_ids.long()).squeeze().float()

    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))

    theta = torch.tensor([theta], dtype=torch.float, device=device_type).expand(
        n_batches
    )

    if tau > 0:
        theta = theta * (1 - torch.exp(-N_b / (n_clusters * tau)) ** 2)

    theta = theta.view(1, -1)

    assert 0 < block_proportion <= 1
    assert correction_method in ["fast", "original"]

    R, E, O, objective_harmony = initialize_centroids(
        Z_norm,
        n_clusters,
        sigma,
        Pr_b,
        Phi,
        theta,
    )

    if verbose:
        print("\tInitialization is completed.")

    for i in range(max_iter_harmony):
        new_objective = clustering(
            Z_norm,
            Pr_b,
            Phi,
            R,
            E,
            O,
            theta,
            tol_clustering,
            max_iter_clustering,
            sigma,
            block_proportion,
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

        if is_convergent_harmony(objective_harmony, new_objective, tol=tol_harmony):
            if verbose:
                print("Reach convergence after {} iteration(s).".format(i + 1))
            break

    return Z_hat


def initialize_centroids(
    Z_norm,
    n_clusters,
    sigma,
    Pr_b,
    Phi,
    theta,
):
    # 1. get centroids from kmeans clustering
    _, Y = kmeans(Z_norm, n_clusters)
    Y_norm = normalize(Y, p=2, dim=1)

    # 2. assign cluster probabilities
    # R = exp(-||Z_i - Y_k||^2 / sigma)
    # because we use cosine distance, we can rewrite it as: R = exp(-2 * (1 - Z_i * Y_k) / sigma)
    R = torch.exp(-2 / sigma * (1 - torch.matmul(Z_norm, Y_norm.t())))
    R = normalize(R, p=1, dim=1)

    # 3. evaluate cluster diversity
    E = torch.matmul(Pr_b, torch.sum(R, dim=0, keepdim=True))
    O = torch.matmul(Phi.t(), R)

    objective_harmony = compute_objective(Y_norm, Z_norm, R, theta, sigma, O, E)
    return R, E, O, objective_harmony


def kmeans(X, n_clusters, tol=1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the K-Means clustering for X with Lloyd method.
    Returns the cluster assignments per row and the cluster centers.

    Code adapted from https://github.com/overshiki/kmeans_pytorch.
    """
    # forgy init: choose initial cluster centers randomly
    indices = np.random.choice(len(X), n_clusters)
    initial_state = X[indices]

    while True:
        dist = torch.cdist(X, initial_state)

        # the closest cluster center for each row
        closest_clusters = torch.argmin(dist, dim=1)

        initial_state_pre = initial_state.clone()

        # calculate the mean of each cluster and treat it as the new cluster center
        for cluster_idx in range(n_clusters):
            rows_with_cluster_idx = torch.nonzero(
                closest_clusters == cluster_idx
            ).squeeze()
            rows_with_cluster = torch.index_select(X, 0, rows_with_cluster_idx)
            initial_state[cluster_idx] = rows_with_cluster.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1))
        )

        if center_shift**2 < tol:
            break

    return closest_clusters, initial_state


def clustering(
    Z_norm,
    Pr_b,
    Phi,
    R,
    E,
    O,
    theta,
    tol,
    max_iter,
    sigma,
    block_proportion,
):
    n_cells = Z_norm.shape[0]

    objectives_clustering = []

    for _ in range(max_iter):
        # Re-compute cluster centroids
        Y = torch.matmul(R.clone().t(), Z_norm)
        Y_norm = normalize(Y, p=2, dim=1)

        # update R
        # update cells in blocks
        update_order_idx = np.arange(n_cells)
        np.random.shuffle(update_order_idx)
        n_blocks = np.ceil(1 / block_proportion).astype(int)
        blocks = np.array_split(update_order_idx, n_blocks)

        for block in blocks:
            R_in = R[block,]
            Phi_in = Phi[block,]

            # Compute O and E on left out data.
            O -= torch.matmul(Phi_in.t(), R_in)
            E -= torch.matmul(Pr_b, torch.sum(R_in, dim=0, keepdim=True))

            # Update and Normalize R
            R_in = torch.exp(
                -2 / sigma * (1 - torch.matmul(Z_norm[block,], Y_norm.t()))
            )
            omega = torch.matmul(Phi_in, torch.pow(torch.div(E + 1, O + 1), theta.t()))
            R_in = R_in * omega
            R_in = normalize(R_in, p=1, dim=1)
            R[block,] = R_in

            # Compute O and E with full data.
            O += torch.matmul(Phi_in.t(), R_in)
            E += torch.matmul(Pr_b, torch.sum(R_in, dim=0, keepdim=True))

        objectives_clustering.append(
            compute_objective(Y_norm, Z_norm, R, theta, sigma, O, E)
        )

        if is_convergent_clustering(objectives_clustering, tol):
            return objectives_clustering[-1]


def correction(X, R, Phi, O, ridge_lambda, correction_method, device_type):
    if correction_method == "fast":
        return correction_fast(X, R, Phi, O, ridge_lambda, device_type)
    else:
        return correction_original(X, R, Phi, ridge_lambda, device_type)


def correction_original(X, R, Phi, ridge_lambda, device_type):
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = torch.cat((torch.ones(n_cells, 1, device=device_type), Phi), dim=1)

    Z = X.clone()
    id_mat = torch.eye(n_batches + 1, n_batches + 1, device=device_type)
    id_mat[0, 0] = 0
    Lambda = ridge_lambda * id_mat
    for k in range(n_clusters):
        Phi_t_diag_R = Phi_1.t() * R[:, k].view(1, -1)
        inv_mat = torch.inverse(torch.matmul(Phi_t_diag_R, Phi_1) + Lambda)
        W = torch.matmul(inv_mat, torch.matmul(Phi_t_diag_R, X))
        W[0, :] = 0
        Z -= torch.matmul(Phi_t_diag_R.t(), W)

    return Z


def correction_fast(X, R, Phi, O, ridge_lambda, device_type):
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = torch.cat((torch.ones(n_cells, 1, device=device_type), Phi), dim=1)

    Z = X.clone()
    P = torch.eye(n_batches + 1, n_batches + 1, device=device_type)
    for k in range(n_clusters):
        O_k = O[:, k]
        N_k = torch.sum(O_k)

        factor = 1 / (O_k + ridge_lambda)
        c = N_k + torch.sum(-factor * O_k**2)
        c_inv = 1 / c

        P[0, 1:] = -factor * O_k

        P_t_B_inv = torch.diag(
            torch.cat(
                (torch.tensor([[c_inv]], device=device_type), factor.view(1, -1)), dim=1
            ).squeeze()
        )
        P_t_B_inv[1:, 0] = P[0, 1:] * c_inv
        inv_mat = torch.matmul(P_t_B_inv, P)

        Phi_t_diag_R = Phi_1.t() * R[:, k].view(1, -1)
        W = torch.matmul(inv_mat, torch.matmul(Phi_t_diag_R, X))
        W[0, :] = 0

        Z -= torch.matmul(Phi_t_diag_R.t(), W)

    return Z


def compute_objective(Y_norm, Z_norm, R, theta, sigma, O, E):
    kmeans_error = torch.sum(R * 2 * (1 - torch.matmul(Z_norm, Y_norm.t())))
    entropy_term = sigma * torch.sum(
        -torch.distributions.Categorical(probs=R).entropy()
    )
    diversity_penalty = sigma * torch.sum(
        torch.matmul(theta, O * torch.log(torch.div(O + 1, E + 1)))
    )
    return kmeans_error + entropy_term + diversity_penalty


def is_convergent_harmony(obj_old, obj_new, tol) -> bool:
    if obj_old is None or obj_new is None:
        return False

    return (obj_old - obj_new) < tol * torch.abs(obj_old)


def is_convergent_clustering(objectives_clustering, tol, window_size=3):
    if len(objectives_clustering) < window_size + 1:
        return False

    obj_old = 0
    obj_new = 0
    for i in range(window_size):
        obj_old += objectives_clustering[-2 - i]
        obj_new += objectives_clustering[-1 - i]

    return (obj_old - obj_new) < tol * torch.abs(obj_old)
