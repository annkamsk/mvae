import torch
import torch.nn.functional as F
from typing import Tuple


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
    X: torch.Tensor,
    batch_ids: torch.Tensor,
    perplexity: float = 30,
):
    """
    Compute the mean of Local Inverse Simpson Index (LISI) for all cells.
    LISI is a measure of the local batch diversity of a dataset.
    For batch_ids including N batches, LISI returns values between 1 and N:
    LISI close to 1 means item is surrounded by neighbors from 1 batch,
    LISI close to N means item is surrounded by neighbors from all N batches.
    """
    # We need at least 3 * n_neighbors to compute the perplexity
    n_neighbors = min(3 * perplexity, X.shape[0] - 1)
    distances, indices = nearest_neighbors(X, n_neighbors)
    n_cells = distances.size(dim=0)
    simpson = torch.zeros(n_cells, device=X.device)

    for i in range(n_cells):
        D_i = distances[i, :]
        Id_i = indices[i, :]
        P_i, H = convert_distance_to_probability(D_i, perplexity)
        simpson[i] = compute_simpson(P_i, H, batch_ids, Id_i)

    return simpson


def nearest_neighbors(
    X: torch.Tensor, n_neighbors: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    pairwise_distances = torch.cdist(X, X.clone())

    # take n_neighbors + 1 because the closest will be the point itself
    distances, indices = pairwise_distances.topk(n_neighbors + 1, largest=False)

    # skip the first one in each row because it is the point itself
    return distances[:, 1:], indices[:, 1:]


def convert_distance_to_probability(
    Di: torch.Tensor, perplexity: float, beta: float = 1.0, tol: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if abs(Hdiff) < tol:
            break

        # If not, increase or decrease precision
        if Hdiff > 0:
            betamin = beta
            if betamax is None:
                beta *= 2
            else:
                beta = (beta + betamax) / 2
        else:
            betamax = beta
            if betamin is None:
                beta /= 2
            else:
                beta = (beta + betamin) / 2

        # Recompute the values
        H, P = compute_entropy(Di, beta)
        Hdiff = H - logU

    return P, H


def compute_entropy(D: torch.Tensor, beta):
    P = torch.exp(-D.clone() * beta)
    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def compute_simpson(
    P: torch.Tensor,
    H: torch.Tensor,
    batch_ids: torch.Tensor,
    indices: torch.Tensor,
    device="cuda",
) -> torch.Tensor:
    """
    Computes Inverse Simpson Index = 1 / Σ p(b), where b is batch
    """
    if H == 0:
        return torch.tensor(-1)

    neighbors_batches = batch_ids[indices].long().squeeze()

    # for each batch, compute the sum of the probabilities of the neighbors
    unique_batches = torch.arange(0, 16, device=device)

    sumP = torch.zeros_like(
        unique_batches, dtype=torch.float, device=device
    ).scatter_add_(0, neighbors_batches, P)

    return 1 / torch.sum(sumP**2)
