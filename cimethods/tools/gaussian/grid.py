from typing import Union
import torch

def ensure_tensor(x, double=False, device=None) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if device is not None:
        x = x.to(device)
    return x

# def f_grid(f, cut_below = 0):
#     x = torch.arange(0, img_size[0]) + 0.5 # value is in the center of pixel
#     y = torch.arange(0, img_size[1]) + 0.5
#     x, y = torch.meshgrid(x, y, indexing='ij') # [..., w, h]
#     val = f(x, y)
#     if cut_below > 0:
#         mask = torch.ones(*f.shape)
#         mask[val < cut_below] = 0
#         val = val*mask
#     return


def gauss_grid(mu_x, 
    mu_y: Union[torch.Tensor, float], 
    s_x: Union[torch.Tensor, float],
    s_y: Union[torch.Tensor, float],
    img_size: tuple[int, int],
    # theta = 0,
    normalize=True,
    cut_below=0,
    torus=False,
    cov=0,  # [0, 1]
    double=False,
    device=None
):
    mu_x = ensure_tensor(mu_x, double, device)
    mu_y = ensure_tensor(mu_y, double, device)
    s_x = ensure_tensor(s_x, double, device)
    s_y = ensure_tensor(s_y, double, device)
    cov = ensure_tensor(cov, double, device)
    sh = mu_x.shape   # TODO: expand to common shape if possible
    assert mu_y.shape == sh, "Size mismatch between tensors"
    assert s_x.shape == sh, "Size mismatch between tensors"
    assert s_y.shape == sh, "Size mismatch between tensors"

    x = torch.arange(0, img_size[0]) + 0.5  # value is in the center of pixel
    y = torch.arange(0, img_size[1]) + 0.5
    x, y = torch.meshgrid(x, y, indexing='ij')  # [..., w, h]
    x = ensure_tensor(x, double, device)
    y = ensure_tensor(y, double, device)

    mu_x = mu_x.unsqueeze(-1).unsqueeze(-1)  # [..., 1-w, 1-h]
    mu_y = mu_y.unsqueeze(-1).unsqueeze(-1)
    s_x = s_x.unsqueeze(-1).unsqueeze(-1)
    s_y = s_y.unsqueeze(-1).unsqueeze(-1)
    cov = cov.unsqueeze(-1).unsqueeze(-1)

    x_shift = x - mu_x
    y_shift = y - mu_y
    if torus:
        x_shift = x_shift + img_size[0] / 2
        x_shift = torch.remainder(x_shift, img_size[0])
        x_shift = x_shift - img_size[0] / 2
        y_shift = y_shift + img_size[1] / 2
        y_shift = torch.remainder(y_shift, img_size[1])
        y_shift = y_shift - img_size[1] / 2

    rel_x = x_shift / s_x
    rel_y = y_shift / s_y
    exp_part = rel_x**2 + rel_y**2
    exp_part -= 2 * cov * rel_x * rel_y
    cov_coef = 1 - cov**2
    exp_part /= 2 * cov_coef

    f = torch.exp(-exp_part)

    if normalize:
        cov_coef = torch.sqrt(cov_coef)
        f = f / (2 * torch.pi * s_x * s_y * cov_coef)

    if cut_below > 0:
        mask = torch.ones(*f.shape)
        mask[f < cut_below] = 0
        f = f * mask

    return f


def rotation_matrix(theta):
    r = torch.zeros(*theta.shape, 2, 2)
    r[..., 0, 0] = r[..., 1, 1] = torch.cos(theta)
    r[..., 1, 0] = torch.sin(theta)
    r[..., 0, 1] = -r[..., 1, 0]
    return r


def gauss_grid_theta(
    mu, 
    sigD, 
    img_size: tuple[int, int],
    theta: Union[int, torch.Tensor]=0,
    normalize=True,
    cut_below=0,
    torus=False,
    double=False,
    device=None
):
    mu = ensure_tensor(mu, double, device)
    sigD = ensure_tensor(sigD, double, device)
    imsize = ensure_tensor(img_size, double, device)
    sh = mu.shape   # TODO: expand to common shape if possible
    assert sh[-1] == 2, "The last dimention of mu and sigma diag expected to be 2"
    assert sigD.shape == sh, "Size mismatch between tensors"
    theta_is_zero = False
    if isinstance(theta, int):
        theta_is_zero = theta == 0
    else:
        theta = ensure_tensor(theta, double, device)
        assert theta.shape == sh[:-1], "Theta batch size is different from mu"
        theta = theta.unsqueeze(-1).unsqueeze(-1)   # [..., 1-w, 1-h]

    x = torch.arange(0, img_size[0], device=device) + 0.5  # value is in the center of pixel
    y = torch.arange(0, img_size[1], device=device) + 0.5
    x, y = torch.meshgrid(x, y, indexing='ij')  # [..., w, h]
    crd = torch.stack([x, y], dim=-1)           # [..., w, h, 2]
    crd = ensure_tensor(crd, double, device)

    mu = mu.unsqueeze(-2).unsqueeze(-2)         # [..., 1-w, 1-h, 2]
    sigD = sigD.unsqueeze(-2).unsqueeze(-2)     # [..., 1-w, 1-h, 2]
    # theta = theta.unsqueeze(-1).unsqueeze(-1)   # [..., 1-w, 1-h]

    if torus:
        crd = crd + imsize / 2
        crd_shift = torch.stack([
            torch.remainder(crd[..., 0] - mu[..., 0], imsize[0]),
            torch.remainder(crd[..., 1] - mu[..., 1], imsize[1]),
        ], dim=-1)
        crd_shift = crd_shift - imsize / 2
    else:
        crd_shift = crd - mu

    if theta_is_zero:
        exp_term = crd_shift / sigD
        exp_term = exp_term**2
        exp_term = exp_term.sum(-1)         # [..., w, h]
    else:
        rot = rotation_matrix(theta)
        rotT = rot.transpose(-1, -2)
        exp_term = torch.matmul(crd_shift.unsqueeze(-2), rot)
        exp_term = exp_term / (sigD**2).unsqueeze(-2)     # <==> matmul( term, diag^-1 )
        exp_term = torch.matmul(exp_term, rotT)
        exp_term = torch.matmul(exp_term, crd_shift.unsqueeze(-1))
        exp_term = exp_term.squeeze(-1).squeeze(-1)

    exp_term = -exp_term / 2
    f = torch.exp(exp_term)

    if normalize:
        if theta_is_zero:
            sig_coef = sigD[..., 0] * sigD[..., 1]
        else:
            sig = torch.matmul(rot * (sigD**2).unsqueeze(-2), rotT)
            sig_norm = sig[..., 0, 0] * sig[..., 1, 1] - sig[..., 0, 1] * sig[..., 1, 0]
            sig_coef = torch.sqrt(sig_norm)
        f = f / (2 * torch.pi * sig_coef)

    if cut_below > 0:
        mask = torch.ones(*f.shape)
        mask[f < cut_below] = 0
        f = f * mask

    return f


# vec_set [..., vec_size (1 + 4), set_size]
# masks [..., set_size]
# images [..., img_h , img_w]
# returns [..., img_h, img_w]
def gauss_grid_from_vec(
    vec_set: torch.Tensor,
    img_size: tuple[int, int],
    masks=None,
    intencities=True,
    **args,
):
    if intencities:
        intensity = vec_set[..., 0, :]
        vec_set = vec_set[..., 1:, :]
    else:
        sh = vec_set.shape
        sh = (*sh[:-2], *sh[-1:])
        intensity = torch.ones(*sh)
    if masks is not None:
        intensity[masks == 0] = 0
    mu_x = vec_set[..., 0, :] * img_size[0]  # [..., set_size]
    mu_y = vec_set[..., 1, :] * img_size[1]
    s_x = vec_set[..., 2, :] + 1
    s_y = vec_set[..., 3, :] + 1

    gausses = gauss_grid(mu_x, mu_y, s_x, s_y, img_size, **args)  # [..., set_size, w, h]
    intensity = intensity.unsqueeze(-1).unsqueeze(-1)
    gausses = intensity * gausses

    return gausses


def vec_set_to_gaussian_spots_sum(vec_set, img_size, **args):
    gausses = gauss_grid_from_vec(vec_set, img_size, **args)
    return gausses.sum(dim=-3)


def model_loss(outputs, images, masks, img_size):
    pred_gauss_images = vec_set_to_gaussian_spots_sum(img_size, outputs, masks)
    pos_corr = (pred_gauss_images * images).view(outputs.shape[0], -1)
    neg_corr = (pred_gauss_images * (1 - images)).view(outputs.shape[0], -1)
    losses = torch.mean(neg_corr, -1) - torch.mean(pos_corr, -1)
    return losses
