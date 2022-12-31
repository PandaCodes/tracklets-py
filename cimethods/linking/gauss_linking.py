from typing import Union

import torch
import torch.nn.functional as F

from tools.gaussian.distance import wasserstein_distance_squared
from tools.gaussian.SpotVector import GaussSpot

class TrackGausses:
    def __init__(self, 
        x: GaussSpot,  # source spots
        y: GaussSpot,  # target spots
        max_dist: float, #  max euclidian  shift allowed
        steps=700,
    ):
        self.maxW2 = max_dist**2
        self.x = x
        self.y = y
        self.steps = steps
        self.dist = None
        
    def calc_distance(self):
        x_dim = GaussSpot(self.x.tensor.unsqueeze(-2))  # ..., n, 1, spot
        y_dim = GaussSpot(self.y.tensor.unsqueeze(-3))  # ..., 1, m, spot
        self.dist = wasserstein_distance_squared(x_dim.mu, y_dim.mu, x_dim.sigD, y_dim.sigD)
        
    
    def optimize(self):
        if self.dist is None:
            print("Distance matrix has not been calculated yet. Please, call calc_dustance() method before")
        n = len(self.x.tensor)
        m = len(self.y.tensor)
        maxW2 = self.maxW2
        dist = self.dist
        x_dim = GaussSpot(self.x.tensor.unsqueeze(-2))  # ..., n, 1, spot
        y_dim = GaussSpot(self.y.tensor.unsqueeze(-3))  # ..., 1, m, spot
        
        # minimize by pi:
        params = torch.zeros(n+1, m+1, requires_grad=True)
        opt = torch.optim.Adam([params], lr=0.1)

        disap = pi = ap = None
        # tkl Dual formulation for the optimal transport
        for e in range(self.steps):
            pi_f = F.softmax(params[1:, :], dim=1).view(n, m+1)
            pi_f = pi_f * x_dim.intensity
            pi_b = F.softmax(params[:, 1:], dim=0).view(n+1, m)
            pi_b = pi_b * y_dim.intensity

            disap = pi_f[:, 0]
            ap = pi_b[0, :]
            pi_f = pi_f[:, 1:]
            pi_b = pi_b[1:, :]
            pi = (pi_f + pi_b)/2
            repr_gap = (pi_f - pi_b)**2

            loss = (dist * pi).sum() + (disap * maxW2).sum() + (ap*maxW2).sum() + 100*repr_gap.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert pi != None
        assert disap != None
        assert ap != None
        pi_ext = torch.zeros(n+1, m+1)
        pi_ext[1:, 1:] = pi
        pi_ext[1:, 0] = disap
        pi_ext[0, 1:] = ap
        
        return pi_ext.detach()
        


def associate(
    transition_matrix: torch.Tensor,
    min_to_max_ratio_boundary=0.3,
    #max_splitting_parts = 3,
):
    assert len(transition_matrix.shape) == 2, "expected transition matrix (2D)"
    assert transition_matrix.max() > 0
    association_matrix = torch.zeros(transition_matrix.shape)
    resid_matrix = transition_matrix.clone()
    noize_removed = False
    while resid_matrix.sum() != 0:
        (values, _) = resid_matrix.max(1, keepdim=True)
        if not noize_removed:
            noize_idx = resid_matrix/values < min_to_max_ratio_boundary
            resid_matrix[noize_idx] = 0
            noize_removed = True
        idxs = values == resid_matrix
        association_matrix[idxs] = resid_matrix[idxs]
        resid_matrix[idxs] = 0
    association_matrix = association_matrix / torch.sum(association_matrix, dim=1, keepdim=True) 
    return association_matrix
