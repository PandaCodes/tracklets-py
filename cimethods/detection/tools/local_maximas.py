import torch
import torch.nn.functional as F

def local_maximas(x, w):
    padding = w // 2
    maxpool = F.max_pool2d(x, w, stride=1, padding=padding)
    return maxpool

def find_local_maximas(
    image: torch.Tensor,
    threshold = 0.001,
    window=5,
):
    assert len(image.shape) == 2, "Expected rectangle-like 2D tensor"
    maximas_mp = local_maximas(image.unsqueeze(0).unsqueeze(0), window).squeeze(0).squeeze(0)
    
    center_idx = torch.logical_and(image == maximas_mp,  maximas_mp > threshold)
    return center_idx

def pix_coords(idx, shape):
    X = torch.arange(0, shape[0]) + 0.0 # to float
    Y = torch.arange(0, shape[1]) + 0.0
    X, Y = torch.meshgrid(X, Y, indexing='ij')
    XY = torch.stack([X, Y], dim=-1)
    return XY[idx]


def filter_close(maximas, d_max = 3.1):
    pdist = torch.pdist(maximas)
    n_centers = len(maximas)
    ftr = torch.ones(n_centers, dtype=torch.bool)
    s = 0
    for j in range(n_centers):
        e = s + (n_centers-1) - j
        if (pdist[s:e] <= d_max).sum() > 0:
            ftr[j] = False
        s = e

    return maximas[ftr]

def filter_close_by_values(centers, center_values, d_max = 3.1):
    center_groups = []
    for val in torch.unique(center_values):
        idx = center_values == val
        ftd_centers = filter_close(centers[idx], d_max=d_max)
        center_groups.append(ftd_centers)
    centers = torch.cat(center_groups)
    return centers