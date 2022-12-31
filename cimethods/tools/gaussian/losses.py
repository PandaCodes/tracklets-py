import scipy
import torch
from .distance import wasserstein_distance_squared


######

def wasserstein_error(predictions, targets):
    # predictions shape (..., s_p , <Spot>);  targets shape :: (..., s_t, <Spot>)
    predictions = predictions.unsqueeze(-2)
    targets = targets.unsqueeze(-3)
    # wst_error shape :: (..., s_p, s_t)
    wst_error = wasserstein_distance_squared(
        predictions.mu, targets.mu, predictions.sigD, targets.sigD)
    wst_error += torch.abs(predictions.i - targets.i)  # intensities
    return wst_error


def wasserstein_hungarian_loss_per_sample(args):
    prediction, target = args
    w_err = wasserstein_error(prediction, target)
    w_err_np = w_err.detach().cpu().numpy()
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(w_err_np)

    loss = w_err[row_idx, col_idx].mean()

    idx = torch.ones(len(prediction), dtype=torch.bool)
    idx[row_idx] = False
    loss += prediction[idx].i.sum()
    return loss


def wasserstein_hungarian_loss(predictions, targets, thread_pool):
    losses = thread_pool.map(wasserstein_hungarian_loss_per_sample, zip(predictions, targets))
    total_loss = torch.mean(torch.stack(losses))
    return total_loss
