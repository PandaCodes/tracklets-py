from typing import Union, TypedDict
import random
from matplotlib.patches import Rectangle
from scipy.stats import truncnorm
import torch

from .utils import as_img_dims, RectangleDimentions
from .SpotVector import GaussSpot


def rand_trunc_gauss(size: Union[tuple[int,...], int], range: tuple[float, float], mu=0., sig=1.):
    # torch.fmod(torch.randn(shape), boundary) # possible approximation
    assert sig != 0, "rand_trunc_gauss: sigma can not be zero"
    range = ( (range[0] - mu) / sig , (range[1] - mu) / sig)
    x = truncnorm.rvs(range[0], range[1], size=size)
    x = x*sig  + mu
    return torch.from_numpy(x)


class TruncGaussParams(TypedDict):
    range: tuple[float, float]
    mu: float
    sig: float


DENSITY = 180


def generate(
    image_size: RectangleDimentions,
    batch_size: Union[int, None]=None,
    intensity=TruncGaussParams(range=(0, 1), mu=1., sig=1.),
    sig=TruncGaussParams(range=(-0.5, 0.5 ), mu=0.5, sig=0.3),
    sig_max_xy_ratio=1.6,
    n_spots: Union[int, None]=None,
    theta=True,
):
    img_dims = as_img_dims(image_size)

    if n_spots is None:
        n_pix = img_dims[0] * img_dims[1]
        n_spots = random.randint(8, n_pix // DENSITY)

    shape = (n_spots,) if batch_size is None else (batch_size, n_spots)
    x = GaussSpot(shape)

    x.sigX = rand_trunc_gauss(shape, **sig)
    ratio_range = (1./sig_max_xy_ratio, sig_max_xy_ratio)
    ratio_mean = (ratio_range[0] + ratio_range[1])/2.
    ratio_sig = (ratio_range[1] - ratio_range[0])/2.
    ratio = rand_trunc_gauss(shape, ratio_range, ratio_mean, ratio_sig)
    x.sigY = x.sigX * ratio

    x.muX = torch.rand(shape) * img_dims[0]
    x.muY = torch.rand(shape) * img_dims[1]

    x.intensity = rand_trunc_gauss(shape, **intensity)
   
    if theta:
        x.theta = torch.randn(shape) * torch.pi #?

    return x


########


def gen_random_vec_set(
    set_size: int,
    vec_size=4,
    batch_size=1,
    intensities=True,
    randomize_count=False
):
    if intensities:
        vec_size += 1
    vec_set = torch.zeros(batch_size, vec_size, set_size)
    masks = torch.zeros(batch_size, set_size)
    for i in range(batch_size):
        count = set_size
        if randomize_count:
            count = random.randint(0, set_size)
        vec_set[i, :, :count] = torch.rand(vec_size, count)
        masks[i, :count] = 1
        # intensities
        if intensities:
            vec_set[i, 0, :count] = rand_trunc_gauss(count, (0, 1), 0.6, 0.25)

    return vec_set, masks