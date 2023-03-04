from dataclasses import dataclass
from typing import List

import torch
from torch.autograd import Variable


@dataclass
class Latent:
    """
    z - data in latent space
    mu - mean of variational posterior
    logvar - log-variance of variational posterior
    """

    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor

    def update(self, latent: "Latent", idx):
        self.z[idx, :] = latent.z
        self.mu[idx, :] = latent.mu
        self.logvar[idx, :] = latent.logvar


def prior_expert(size: List[int], use_cuda=False):
    """
    Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    size - dimensionality of Gaussian
    """
    # FIXME replace with torch.zeros(size, requires_grad=True)
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


def initialize_latent(size: List[int], use_cuda=False) -> Latent:
    """
    Initializes arrays of mu, logvar, and z.
    n_samples - number of samples
    n_latent - dimensionality of latent space
    """
    mu, logvar = prior_expert(size, use_cuda)
    z = sample_latent(mu, logvar, use_cuda)
    return Latent(z, mu, logvar)


def sample_latent(mu, logvar, use_cuda=False) -> torch.Tensor:
    std = logvar.mul(0.5).exp_()
    eps = torch.FloatTensor(std.size()).normal_()
    if use_cuda:
        eps = eps.to(torch.device("cuda"))
    eps = eps.mul_(std).add_(mu)
    return eps
