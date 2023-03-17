"""
Includes code from https://github.com/slowkow/harmonypy/ adjusted to work with Tensors.
"""

from src.types import ModelInputT, ModelOutputT
from src.harmony import harmonize
import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn


class LISILoss(nn.Module):
    def __init__(self, perplexity: float = 30, tol: float = 1e-5) -> None:
        super().__init__()
        self.perplexity = perplexity
        self.tol = tol

    def forward(self, input: ModelInputT, output: ModelOutputT, device="cuda"):
        batch_id = input["batch_id1"]
        poe = output["poe_latent"]["z"]

        poe_corrected = harmonize(poe, batch_id, device_type=device)

        return compute_lisi(poe_corrected, batch_id, self.perplexity, device)


def compute_lisi(
    X: torch.Tensor,
    batch_ids: torch.Tensor,
    perplexity: float = 30,
    device="cuda",
):
    """
    Compute the Local Inverse Simpson Index (LISI) for batch_ids. LISI is a measure
    of the local batch diversity of a dataset.
    For batch_ids including N batches:
        - If LISI is approximately equal to N for an item in the data matrix,
          that means that the item is surrounded by neighbors from all N
          batches.
        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 batch.
    """
    # We need at least 3 * n_neigbhors to compute the perplexity
    knn_model = NearestNeighbors(n_neighbors=perplexity * 3, algorithm="kd_tree")
    if device == "cpu":
        knn = knn_model.fit(X)
        distances, indices = knn.kneighbors(X)
    else:
        knn = knn_model.fit(X.cpu())
        distances, indices = knn.kneighbors(X.cpu())

    # Don't count yourself
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    categories = torch.unique(batch_ids)
    n_categories = len(categories.size(dim=0))

    distances = torch.tensor(distances, dtype=torch.float, device=device)

    simpson = compute_simpson(
        distances.T, indices.T, categories, n_categories, perplexity
    )
    return 1 / simpson


def compute_simpson(
    distances: torch.Tensor,
    indices: np.ndarray,
    categories: torch.Tensor,
    perplexity: float,
    tol: float = 1e-5,
):
    n_cells = distances.size(dim=1)
    simpson = torch.zeros(n_cells)

    # Loop through each cell.
    for i in range(n_cells):
        # Compute Hdiff
        H, P = compute_hdiff(distances, i, perplexity, tol)

        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in categories:
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = torch.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson


def compute_hdiff(distances: torch.Tensor, i, perplexity, tol):
    beta = 1
    betamin = -np.inf
    betamax = np.inf
    logU = torch.log(perplexity)

    P = torch.exp(-distances[:, i] * beta)
    P_sum = torch.sum(P)
    if P_sum == 0:
        H = 0
        P = torch.zeros(distances.size(dim=0))
    else:
        H = torch.log(P_sum) + beta * torch.sum(distances[:, i] * P) / P_sum
        P = P / P_sum

    Hdiff = H - logU
    n_tries = 50
    for _ in range(n_tries):
        # Stop when we reach the tolerance
        if abs(Hdiff) < tol:
            return H, P

        # Update beta
        if Hdiff > 0:
            betamin = beta
            if not np.isfinite(betamax):
                beta *= 2
            else:
                beta = (beta + betamax) / 2
        else:
            betamax = beta
            if not np.isfinite(betamin):
                beta /= 2
            else:
                beta = (beta + betamin) / 2

        # Compute Hdiff
        P = torch.exp(-distances[:, i] * beta)
        P_sum = torch.sum(P)
        if P_sum == 0:
            H = 0
            P = torch.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P = P / P_sum

        Hdiff = H - logU
    return H, P
