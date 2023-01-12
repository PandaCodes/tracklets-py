import torch
import torch.nn.functional as F


nbh_kernel = torch.FloatTensor([[[
    [0.75, 1, 0.75],
    [1,    0, 1   ],
    [0.75, 1, 0.75]
]]])

def neighbours_loss(x):
    neighbours = F.conv2d(x, nbh_kernel, stride=1, padding=1)
    return (neighbours*x.detach()).mean()

def max_pool_diff_loss(x):
    maxpool = F.max_pool2d(x, 3, stride=1, padding=1)
    non_max_idx = x < maxpool
    return 0 if non_max_idx.sum() == 0 else x[non_max_idx].mean()