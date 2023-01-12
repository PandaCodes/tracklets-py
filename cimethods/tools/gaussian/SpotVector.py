from typing import Union

import torch
from .grid import gauss_grid_theta
from .utils import RectangleDimentions, as_img_dims, real_to_range



class TensorHolder():
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def __getitem__(self, indexes):
        return self.__class__(self.tensor[indexes])
    
    def __len__(self):
        return len(self._tensor)

    def clone(self):
        return self.__class__(self.tensor.clone())

    _tensor: torch.Tensor
    @property
    def tensor(self):
        return self._tensor
    
    def to(self, device): # type?
        self._tensor = self._tensor.to(device)
        return self
    


class GaussSpot(TensorHolder):

    SIZE = 6 # = 2 x mu + 2 x sig + theta + intensity
    
    def __init__(self, arg: Union[tuple[int,...], torch.Tensor ]):
        if isinstance(arg, tuple):
            super().__init__(torch.zeros(*arg, GaussSpot.SIZE))
        else:
            #assert len(arg.shape) > 1
            assert arg.shape[-1] == GaussSpot.SIZE, "GaussSpot tensor last dimention expected to be certain size"
            super().__init__(arg)

    @property
    def intensity(self):
        return self.tensor[..., -1]
    @intensity.setter
    def intensity(self, value, idx=None):
        if idx is None:
            self.tensor[..., -1] = value
        else:
            self.tensor[idx, -1] = value

    @property
    def muX(self):
        return self.tensor[..., 0]
    @muX.setter
    def muX(self, val):
        self.tensor[..., 0] = val

    @property
    def muY(self):
        return self.tensor[..., 1]
    @muY.setter
    def muY(self, val):
        self.tensor[..., 1] = val

    @property
    def mu(self):
        return self.tensor[..., :2]
    @mu.setter
    def mu(self, val):
        self.tensor[..., :2] = val

    @property
    def sigX(self):
        return self.tensor[..., 2]
    @sigX.setter
    def sigX(self, val):
        self.tensor[..., 2] = val

    @property
    def sigY(self):
        return self.tensor[..., 3]
    @sigY.setter
    def sigY(self, val):
        self.tensor[..., 3] = val

    # diagonal
    @property
    def sigD(self):
        return self.tensor[..., 2:4]
    @sigD.setter
    def sigD(self, value, idx=None):
        if idx is None:
            self.tensor[..., 2:4] = value
        else:
            self.tensor[idx, 2:4] = value

    @property
    def theta(self):
        return self.tensor[..., 4]
    @theta.setter
    def theta(self, value):
        self.tensor[..., 4] = value

    @property
    def rotation_matrix(self):
        r = torch.zeros(*self.tensor.shape[:-1], 2, 2)
        r[..., 0, 0] = r[..., 1, 1] = torch.cos(self.theta)
        r[..., 1, 0] = torch.sin(self.theta)
        r[..., 0, 1] = -r[..., 1, 0]
        return r

    @property
    def sig(self):
        r = self.rotation_matrix
        return torch.matmul(r * self.sigD, r.transpose(-1, -2))

    ######


    def gauss_grid(self, size: RectangleDimentions, **args):
        shape = as_img_dims(size)
        # return gauss_grid(self.muX, self.muY, self.sigX, self.sigY, shape, **args)
        return gauss_grid_theta(self.mu, self.sigD, shape, **args, theta=self.theta)

    def sum_image(self, size: RectangleDimentions, dim=-3, clamp=True, cut_below=0., **kwargs):
        grid = self.gauss_grid(size, cut_below=cut_below, torus=True, **kwargs)
        grid = grid * self.intensity.unsqueeze(-1).unsqueeze(-1)
        img = grid.sum(dim)
        if clamp:
            img = img.clamp(0, 1)
        return img

    # def wasserstein_distance_2(self, other: DiagGaussSpotTensor):
    #     d = (self.mu - other.mu)**2
    #     d = d.sum(-1)

    #     sig_1 = silf.sig

    #     u_0 = self.rotation_matrix
    #     sig_0_sqrt = torch.mathmul(u_0 * torch.sqrt(self.sigD), u_0.transpose(-1, -2))
    #     sub_term = torch.mathmul(sig_0_sqrt, sig_1)
    #     sub_term = torch.mathmul(sub_term, sig_0_sqrt)

    #     sig_d = sig0 + sig1 - 2 * torch.sqrt(sig0 * sig1)
    #     sig_d = sig_d.sum(-1)

    def limit_values(self,
        size: RectangleDimentions,
        sig: tuple[float, float], 
        intensity:  tuple[float, float], 
        torus=False,
        clamp=False,
        in_place=False,
    ):
        shape = as_img_dims(size)
        gs = self if in_place else GaussSpot(self.tensor.shape[:-1])
        gs.theta = self.theta
        if clamp:
            gs.muX = self.muX.clamp(0, shape[0])
            gs.muY = self.muY.clamp(0, shape[1])
            gs.intensity = self.intensity.clamp(*intensity)
            gs.sigD = self.sigD.clamp(*sig)
            return gs

        if torus:
            gs.muX = torch.remainder(self.muX, shape[0])
            gs.muY = torch.remainder(self.muY, shape[1])
        else:
            gs.muX = real_to_range(self.muX, 0, shape[0])
            gs.muY = real_to_range(self.muY, 0, shape[1])
        gs.intensity = real_to_range(self.intensity, *intensity)
        gs.sigD = real_to_range(self.sigD, *sig)
        return gs
