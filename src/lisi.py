from typing import Tuple
from src.types import ModelInputT, ModelOutputT
from src.harmony import harmonize

import torch


def batch_integration_loss(
    input: ModelInputT,
    output: ModelOutputT,
    perplexity: float = 30,
    tol: float = 1e-5,
    device="cuda",
):
    """
    Tries to correct the POE latent space for batch effects with Harmony and calculates loss
    as LISI (Local Inverse Simpson Index) score.
    """
    batch_id = input["batch_id1"]
    poe = output["poe_latent"]["z"]

    poe_corrected = harmonize(poe, batch_id, device_type=device)
    return compute_lisi(poe_corrected, batch_id, perplexity, device)


def compute_lisi(
    X: torch.Tensor,
    batch_ids: torch.Tensor,
    perplexity: float = 30,
):
    """
    Compute the Local Inverse Simpson Index (LISI). LISI is a measure
    of the local batch diversity of a dataset.
    For batch_ids including N batches, LISI returns values between 1 and N:
    LISI close to 1 means item is surrounded by neighbors from 1 batch,
    LISI close to N means item is surrounded by neighbors from all N batches.
    """
    # We need at least 3 * n_neigbhors to compute the perplexity
    distances, indices = nearest_neighbors(X, perplexity * 3)

    n_cells = distances.size(dim=0)
    simpson = torch.zeros(n_cells)

    for i in range(n_cells):
        D_i = distances[i, :]
        Id_i = indices[i, :]
        P_i, H = convert_distance_to_probability(D_i, perplexity)
        simpson[i] = compute_simpson(P_i, H, batch_ids, Id_i)

    return simpson


def nearest_neighbors(
    X: torch.Tensor, n_neighbors: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    pairwise_distances = torch.cdist(X, X)

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
    Perplexity (how well the probability distribution predicts the distances) is a hyperparameter defined
    as 2^H(P_ij) where H is the Shannon entropy of the distribution.

    Theory described in: https://www.researchgate.net/publication/228339739_Viualizing_data_using_t-SNE
    """
    betamin = None
    betamax = None

    logU = torch.log(torch.tensor([perplexity]))

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
    P = torch.exp(-D * beta)
    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def compute_simpson(
    P: torch.Tensor,
    H: torch.Tensor,
    batch_ids: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if H == 0:
        return torch.tensor(-1)

    # Inverse Simpson Index = 1 / Σ p(b) where b is batch
    neighbors_batches = batch_ids[indices]

    # for each batch, compute the sum of the probabilities of the neighbors
    unique_batches = neighbors_batches.unique(dim=0)
    sumP = torch.zeros_like(unique_batches, dtype=torch.float).scatter_add_(
        0, neighbors_batches, P
    )

    return 1 / torch.sum(sumP**2)
