"""
Methods defined in this file are adapted from the following repository:
https://github.com/lilab-bcb/harmony-pytorch
"""

import torch

import numpy as np


from sklearn.cluster import KMeans
from torch.nn.functional import normalize


def initialize_centroids(
    Z_norm,
    n_clusters,
    sigma,
    Pr_b,
    Phi,
    theta,
    random_state,
    device_type,
    n_jobs,
    n_init=10,
):
    n_cells = Z_norm.shape[0]

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

    # Initialize R
    R = torch.exp(-2 / sigma * (1 - torch.matmul(Z_norm, Y_norm.t())))
    R = normalize(R, p=1, dim=1)

    E = torch.matmul(Pr_b, torch.sum(R, dim=0, keepdim=True))
    O = torch.matmul(Phi.t(), R)

    objectives_harmony = []
    compute_objective(
        Y_norm, Z_norm, R, theta, sigma, O, E, objectives_harmony, device_type
    )

    return R, E, O, objectives_harmony


def clustering(
    Z_norm,
    Pr_b,
    Phi,
    R,
    E,
    O,
    n_clusters,
    theta,
    tol,
    objectives_harmony,
    max_iter,
    sigma,
    block_proportion,
    device_type,
    n_init=10,
):
    n_cells = Z_norm.shape[0]

    objectives_clustering = []

    for i in range(max_iter):
        # Compute Cluster Centroids
        Y = torch.matmul(R.t(), Z_norm)
        Y_norm = normalize(Y, p=2, dim=1)

        idx_list = np.arange(n_cells)
        np.random.shuffle(idx_list)
        block_size = int(n_cells * block_proportion)
        pos = 0
        while pos < len(idx_list):
            idx_in = idx_list[pos : (pos + block_size)]
            R_in = R[idx_in,]
            Phi_in = Phi[idx_in,]

            # Compute O and E on left out data.
            O -= torch.matmul(Phi_in.t(), R_in)
            E -= torch.matmul(Pr_b, torch.sum(R_in, dim=0, keepdim=True))

            # Update and Normalize R
            R_in = torch.exp(
                -2 / sigma * (1 - torch.matmul(Z_norm[idx_in,], Y_norm.t()))
            )
            omega = torch.matmul(Phi_in, torch.pow(torch.div(E + 1, O + 1), theta.t()))
            R_in = R_in * omega
            R_in = normalize(R_in, p=1, dim=1)
            R[idx_in,] = R_in

            # Compute O and E with full data.
            O += torch.matmul(Phi_in.t(), R_in)
            E += torch.matmul(Pr_b, torch.sum(R_in, dim=0, keepdim=True))

            pos += block_size

        compute_objective(
            Y_norm, Z_norm, R, theta, sigma, O, E, objectives_clustering, device_type
        )

        if is_convergent_clustering(objectives_clustering, tol):
            objectives_harmony.append(objectives_clustering[-1])
            break


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


def compute_objective(
    Y_norm, Z_norm, R, theta, sigma, O, E, objective_arr, device_type
):
    kmeans_error = torch.sum(R * 2 * (1 - torch.matmul(Z_norm, Y_norm.t())))
    entropy_term = sigma * torch.sum(
        -torch.distributions.Categorical(probs=R).entropy()
    )
    diversity_penalty = sigma * torch.sum(
        torch.matmul(theta, O * torch.log(torch.div(O + 1, E + 1)))
    )
    objective = kmeans_error + entropy_term + diversity_penalty

    objective_arr.append(objective)


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
