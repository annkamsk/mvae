"""
Methods defined in this file are adapted from the following repository:
https://github.com/lilab-bcb/harmony-pytorch
"""

import torch

import numpy as np

from sklearn.cluster import KMeans
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
    correction_method: str = "fast",
    random_state: int = 0,
    device_type: str = "cpu",
    n_jobs: int = -1,
    verbose: bool = True,
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
    ``torch.Tensor``
        The integrated embedding by Harmony, of the same shape as the input embedding.
    """
    Z = X.to(dtype=torch.float)
    Z_norm = normalize(Z, p=2, dim=1)
    n_cells = Z.shape[0]

    # number of items in each batch
    _, N_b = torch.unique(batch_ids, sorted=True, return_counts=True)
    # proportion of items in each batch
    Pr_b = N_b.view(-1, 1) / n_cells

    n_batches = N_b.size(dim=0)

    Phi = one_hot(batch_ids)

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

    np.random.seed(random_state)

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
        objective = clustering(
            Z_norm,
            Pr_b,
            Phi,
            R,
            E,
            O,
            n_clusters,
            theta,
            tol_clustering,
            max_iter_clustering,
            sigma,
            block_proportion,
            device_type,
        )
        objectives_harmony.append(objective)

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
        return Z_hat
    else:
        return Z_hat


def initialize_centroids(
    Z_norm,
    n_clusters,
    sigma,
    Pr_b,
    Phi,
    theta,
    random_state,
    device_type,
    n_init=10,
):
    kmeans_params = {
        "n_clusters": n_clusters,
        "init": "k-means++",
        "n_init": n_init,
        "random_state": random_state,
        "max_iter": 25,
    }

    kmeans = KMeans(**kmeans_params)

    if device_type == "cpu":
        kmeans.fit(Z_norm)
    else:
        kmeans.fit(Z_norm.cpu())

    Y = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device_type)
    Y_norm = normalize(Y, p=2, dim=1)

    # assign cluster probabilities
    R = torch.exp(-2 / sigma * (1 - torch.matmul(Z_norm, Y_norm.t())))
    R = normalize(R, p=1, dim=1)

    # batch diversity statistics
    E = torch.matmul(Pr_b, torch.sum(R, dim=0, keepdim=True))
    O = torch.matmul(Phi.t(), R)

    objective_harmony = compute_objective(Y_norm, Z_norm, R, theta, sigma, O, E)
    return R, E, O, [objective_harmony]


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
        # Compute Cluster Centroids
        Y = torch.matmul(R.t(), Z_norm)
        Y_norm = normalize(Y, p=2, dim=1)

        # Update cells in blocks
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


def is_convergent_harmony(objectives_harmony, tol):
    if len(objectives_harmony) < 2:
        return False

    obj_old = objectives_harmony[-2]
    obj_new = objectives_harmony[-1]

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
