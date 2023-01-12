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
    assert len(image.shape) >= 2, "Expected tensor to be at least 2D"
    assert len(image.shape) <= 4, "Expected tensor to have at max 4 dimentions (B, C, W, H)"
    
    need_unsqueeze_batch = False
    if len(image.shape) == 2:
        need_unsqueeze_batch = True
        image = image.unsqueeze(0)
    need_squeeze_chan = False
    if len(image.shape) == 3:
        need_squeeze_chan = True
        image = image.unsqueeze(1) # add chan dim
    
    maximas_mp = local_maximas(image, window)
    center_idx = torch.logical_and(image == maximas_mp,  maximas_mp > threshold)

    if need_squeeze_chan:
        center_idx = center_idx.squeeze(1)
    if need_unsqueeze_batch:
        center_idx = center_idx.squeeze(0)
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