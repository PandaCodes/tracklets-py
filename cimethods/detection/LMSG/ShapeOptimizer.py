from typing import Union

import torch
import torch.nn.functional as F

from tools.gaussian.gen import generate, TruncGaussParams
from ..tools.local_maximas import find_local_maximas, pix_coords, filter_close_by_values

class ShapeOptimizer():
    def __init__(self, image: torch.Tensor):
        self.image = image
        self.init_gen_opts = {
            "sig": TruncGaussParams(range=(0.6, 2), mu=0.65, sig=0.7),
            "intensity": TruncGaussParams(range=(0, 20), mu=1, sig=1),
        }
        self._spots = None

    @property
    def spots(self):
        assert self._spots is not None, "Call init_spots() first"
        # TODO: localise limit_values in one place: ValueLimiter interface, or something like that 
        #           - to have one limit-values-init per Spot instance
        return self._spots.limit_values(
            self.image.shape, 
            sig=self.init_gen_opts["sig"]["range"],
            intensity=self.init_gen_opts["intensity"]["range"],
        )
    def get_centers(self):
        return self.spots.mu

    def init_spots(self, threshold=0.00):
        # TODO: separate method from this class (suuply with ready-made spot centers)
        image = self.image
        centers_idx = find_local_maximas(image, threshold=threshold, window=5)
        centers = pix_coords(centers_idx, image.shape)
        center_values = image[centers_idx]
        centers = filter_close_by_values(centers, center_values, d_max=3)
        centers += 0.5 # pixel centers

        n_spots = len(centers)  
        spots = generate(
            n_spots=n_spots, 
            image_size=image.shape, 
            **self.init_gen_opts, 
        )
        
        spots.mu = centers
        spots.tensor.requires_grad_()
        
        self._spots = spots
    
    def spots_image(self):
        # TODO: shape out meaning of `cut_below` parameter
        spots = self.spots
        return spots[spots.intensity > 0].sum_image(self.image.shape, cut_below=0.03, normalize=False)
    
    def optimize(self, lr=0.5, avg_loss_stop = 1e-50, avg_loss_stop_n=5):
        image = self.image
        assert self._spots is not None, "Call init_spots() first"
        optimizer = torch.optim.Adam(
            [self._spots.tensor], lr=lr
        )

        torch.set_grad_enabled(True)
        
        last_losses = torch.ones(avg_loss_stop_n)*torch.inf
        for i in range(100000):
            predicted = self.spots.sum_image(image.shape, cut_below=0.03, normalize=False)

            loss = F.mse_loss(predicted, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

            loss_val = loss.item()
            if (last_losses - loss_val).mean() < avg_loss_stop:
                print("early stop", i)
                break
            last_losses[i%avg_loss_stop_n] = loss_val
        
        torch.set_grad_enabled(False)
        return last_losses

        