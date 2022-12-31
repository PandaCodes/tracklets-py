import torch


def wasserstein_distance_squared(
    m0: torch.Tensor,
    m1: torch.Tensor,
    sig0: torch.Tensor, 
    sig1: torch.Tensor,
    diag=True
):

    if diag:
        # diagonal case only here TODO: common  matrix case
        sig_d = sig0 + sig1 - 2 * torch.sqrt(sig0 * sig1)
        sig_d = sig_d.sum(-1)
    else:
        raise NotImplementedError("Wasserstein distance with full sigmas not implemented")

    d = sig_d + ((m0 - m1)**2).sum(-1)
    return d


def wasserstein_distance(*args):   # how to make types following??
    wd = wasserstein_distance_squared(*args)
    return torch.sqrt(wd)

