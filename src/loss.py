import numpy as np
import pandas as pd
from scipy.stats import chi2
import torch
import torch.nn.functional as F
from typing import Tuple
from mudata import MuData


def get_loss_fun(loss_fun: str):
    return {
        "mse": mse,
        "bce": bce,
    }[loss_fun]


def mse(pred: torch.Tensor, real: torch.Tensor, dropout=False) -> torch.Tensor:
    if not dropout:
        return F.mse_loss(pred, real, reduction="mean")

    dropout_mask = create_dropout_mask(real)
    return torch.sum((real - pred).pow(2) * dropout_mask) / torch.sum(dropout_mask)


def bce(pred: torch.Tensor, real: torch.Tensor, dropout=False) -> torch.Tensor:
    if not dropout:
        return torch.nn.BCEWithLogitsLoss(reduce="mean")(pred, real)

    dropout_mask = create_dropout_mask(real)
    n_nonzero_features = dropout_mask.sum().int()
    class_weight = (
        dropout_mask.shape[0] * dropout_mask.shape[1] - n_nonzero_features
    ) / n_nonzero_features
    return torch.nn.BCEWithLogitsLoss(reduce="mean", pos_weight=class_weight)(
        pred, real
    )


def create_dropout_mask(real: torch.Tensor) -> torch.Tensor:
    dropout_mask = (real != 0).float()
    n_nonzero_features = dropout_mask.sum().int()
    mask_size = dropout_mask.size()
    dropout_mask = dropout_mask.reshape(-1).clone()
    dropout_mask[torch.randperm(len(dropout_mask))[:n_nonzero_features]] = 1
    return dropout_mask.reshape(mask_size)


def mmd(x: torch.Tensor, y: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """
    Computes MMD between x and y.
    """
    x_kernel = gaussian_kernel_matrix(x, x, alphas)
    y_kernel = gaussian_kernel_matrix(y, y, alphas)
    xy_kernel = gaussian_kernel_matrix(y, x, alphas)

    loss = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return loss


def gaussian_kernel_matrix(
    x: torch.Tensor, y: torch.Tensor, alphas: torch.Tensor
) -> torch.Tensor:
    """
    Computes multiscale-RBF kernel between x and y.
    """
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1.0 / (2.0 * alphas)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    return torch.transpose(output, 0, 1)


def compute_lisi(
    neighbors: Tuple[torch.Tensor, torch.Tensor],
    batch_ids: torch.Tensor,
    batch_n: int,
    perplexity: float = 30,
    summary_writer=None,
):
    """
    Compute the mean of Local Inverse Simpson Index (LISI) for all cells.
    LISI is a measure of the local batch diversity of a dataset.
    For batch_ids including N batches, LISI returns values between 1 and N:
    LISI close to 1 means item is surrounded by neighbors from 1 batch,
    LISI close to N means item is surrounded by neighbors from all N batches.
    """
    distances, indices = neighbors
    n_cells = distances.size(dim=0)
    simpson = torch.zeros(n_cells, device=batch_ids.device)

    for i in range(n_cells):
        D_i = distances[i, ~torch.isnan(distances[i, :])]
        Id_i = indices[i, ~torch.isnan(indices[i, :])].long()
        P_i, H = convert_distance_to_probability(D_i, perplexity)
        simpson[i] = compute_simpson(P_i, batch_ids, Id_i, batch_n)

    return simpson


def nearest_neighbors(
    X: torch.Tensor, n_neighbors: int = 5, mutual: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    pairwise_distances = torch.cdist(X, X.clone())

    # take n_neighbors + 1 because the closest will be the point itself
    distances_all, indices_all = pairwise_distances.topk(n_neighbors + 1, largest=False)

    # skip the first one in each row because it is the point itself
    distances, indices = distances_all[:, 1:], indices_all[:, 1:]

    if not mutual:
        return distances, indices

    # filter only mutual nearest neighbors
    mnn_indices = torch.full(indices.size(), device=X.device, fill_value=torch.nan)
    mnn_distances = torch.full(distances.size(), device=X.device, fill_value=torch.nan)
    for cell_idx in range(indices.size(0)):
        neighbors = indices[cell_idx]
        is_neighbor_of = (
            (indices.flatten() == cell_idx)
            .nonzero()
            .div(indices.size(1), rounding_mode="floor")
        )
        mutual_idx = (neighbors == is_neighbor_of).any(dim=0)
        if not mutual_idx.any():
            continue
        mutual = neighbors[mutual_idx]
        mutual_distances = distances[cell_idx, mutual_idx]
        mnn_indices[cell_idx][: len(mutual)] = mutual
        mnn_distances[cell_idx][: len(mutual)] = mutual_distances
    return mnn_distances, mnn_indices


def convert_distance_to_probability(
    Di: torch.Tensor, perplexity: float, beta: float = 1.0, tol: float = 1e-5
) -> torch.Tensor:
    """
    Computes Gaussian kernel-based distribution of a cell neighborhood by
    converting distances into conditional probabilities P_ij.

    P_ij = probability that point x_i would pick x_j as a neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at x_i

    Performs binary search for probability P_ij for given i that's within tolerance tol of perplexity.
    Perplexity (how well the probability distribution predicts the distances) is a hyperparameter defined as 2^H(P_ij) where H is the Shannon entropy of the distribution.

    Theory described in: https://www.researchgate.net/publication/228339739_Viualizing_data_using_t-SNE
    """
    betamin = None
    betamax = None

    logU = torch.log(
        torch.tensor(
            [perplexity], dtype=torch.float, device=Di.device, requires_grad=True
        )
    )

    H, P = compute_entropy(Di, beta)
    Hdiff = H - logU

    n_tries = 50
    for _ in range(n_tries):
        # Is the perplexity within tolerance
        if abs(Hdiff) <= tol:
            break

        # If not, increase or decrease precision
        if Hdiff > 0:
            betamin = beta
            if betamax is None:
                beta *= 2.0
            else:
                beta = (beta + betamax) / 2.0
        else:
            betamax = beta
            if betamin is None:
                beta /= 2.0
            else:
                beta = (beta + betamin) / 2.0

        # Recompute the values
        H, P = compute_entropy(Di, beta)
        Hdiff = H - logU

    return P, H


def compute_entropy(D: torch.Tensor, beta):
    P = torch.exp(-D.clone() * beta)
    sumP = torch.nansum(P)

    if sumP == 0:
        return torch.zeros(1, device=D.device), torch.zeros(D.shape, device=D.device)

    H = torch.log(sumP) + beta * torch.nansum(D * P) / sumP
    P = P / sumP

    return H, P


def compute_simpson(
    P: torch.Tensor,
    batch_ids: torch.Tensor,
    indices: torch.Tensor,
    batch_n: int = 16,
    device="cuda",
) -> torch.Tensor:
    """
    Computes Inverse Simpson Index = 1 / Σ p(b), where b is batch
    """
    neighbors_batches = batch_ids[indices].long().view(-1)

    # for each batch, compute the sum of the probabilities of the neighbors
    unique_batches = torch.arange(0, batch_n, device=device)

    sumP = torch.zeros_like(
        unique_batches, dtype=torch.float, device=device
    ).scatter_add_(0, neighbors_batches, P)

    return 1 / torch.nansum(sumP**2)


def compute_spatial_loss(
    neighbors_indices: torch.Tensor, neighbors_prior: torch.Tensor
):
    """
    Calculate how many neighbors in latent are also spatially close.
    neighbors_prior: tensor of shape (n_cells, prior_n_neighbors)
    cell_id: tensor of shape (n_cells) with cell index
    neighbor_indices: tensor of shape (n_cells, n_neighbors) with indices of neighbors in latent space
    """
    spatial_loss = 0
    for i in range(neighbors_indices.size(0)):
        neigh_cat, counts = torch.cat(
            [neighbors_indices[i], neighbors_prior[i]]
        ).unique(return_counts=True)
        spatial_loss += neigh_cat[torch.where(counts.gt(1))].count_nonzero(0)

    return 1 / spatial_loss


def calc_kBET(
    data: MuData,
    batch_key: str,
    rep: str = "pca",
    K: int = 25,
    alpha: float = 0.05,
    n_jobs: int = -1,
    random_state: int = 0,
    temp_folder: str = None,
    use_cache: bool = True,
) -> Tuple[float, float, float]:
    """Calculate the kBET metric of the data regarding a specific sample attribute and embedding.
    https://github.com/lilab-bcb/pegasus
    The kBET metric is defined in [Büttner18]_, which measures if cells from different samples mix well in their local neighborhood.

    attr: ``str``
        The sample attribute to consider. Must exist in ``data.obs``.

    rep: ``str``, optional, default: ``"pca"``
        The embedding representation to be used. The key ``'X_' + rep`` must exist in ``data.obsm``. By default, use PCA coordinates.

    K: ``int``, optional, default: ``25``
        Number of nearest neighbors, using L2 metric.

    alpha: ``float``, optional, default: ``0.05``
        Acceptance rate threshold. A cell is accepted if its kBET p-value is greater than or equal to ``alpha``.

    Returns
    -------
    stat_mean: ``float``
        Mean kBET chi-square statistic over all cells.

    pvalue_mean: ``float``
        Mean kBET p-value over all cells.

    accept_rate: ``float``
        kBET Acceptance rate of the sample.
    """
    ideal_dist = (
        data.obs[batch_key].value_counts(normalize=True, sort=False).values
    )  # ideal no batch effect distribution
    nsample = data.shape[0]
    nbatch = ideal_dist.size

    attr_values = data.obs[batch_key].values.copy()

    X = torch.tensor(data.obsm[f"{rep}_z"], device="cuda")

    _, indices = nearest_neighbors(X, K)
    indices_np = indices.cpu().numpy().astype(int)

    knn_indices = np.concatenate(
        (np.arange(nsample).reshape(-1, 1), indices_np[:, 0 : K - 1]), axis=1
    )  # add query as 1-nn

    # partition into chunks
    n_jobs = min(16, nsample)
    starts = np.zeros(n_jobs + 1, dtype=int)
    quotient = nsample // n_jobs
    remainder = nsample % n_jobs
    for i in range(n_jobs):
        starts[i + 1] = starts[i] + quotient + (1 if i < remainder else 0)

    from joblib import Parallel, delayed, parallel_backend

    with parallel_backend("loky", inner_max_num_threads=1):
        kBET_arr = np.concatenate(
            Parallel(n_jobs=n_jobs, temp_folder=temp_folder)(
                delayed(calc_kBET_for_one_chunk)(
                    knn_indices[starts[i] : starts[i + 1], :],
                    attr_values,
                    ideal_dist,
                    K,
                )
                for i in range(n_jobs)
            )
        )

    res = kBET_arr.mean(axis=0)
    stat_mean = res[0]
    pvalue_mean = res[1]
    accept_rate = (kBET_arr[:, 1] >= alpha).sum() / nsample

    return (stat_mean, pvalue_mean, accept_rate)


def calc_kBET_for_one_chunk(knn_indices, attr_values, ideal_dist, K):
    dof = ideal_dist.size - 1

    ns = knn_indices.shape[0]
    results = np.zeros((ns, 2))
    for i in range(ns):
        observed_counts = (
            pd.Series(attr_values[knn_indices[i, :]]).value_counts(sort=False).values
        )
        expected_counts = ideal_dist * K
        stat = np.sum(
            np.divide(
                np.square(np.subtract(observed_counts, expected_counts)),
                expected_counts,
            )
        )
        p_value = 1 - chi2.cdf(stat, dof)
        results[i, 0] = stat
        results[i, 1] = p_value

    return results
