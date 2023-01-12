from typing import Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from detection.tools.local_maximas import find_local_maximas
from tools.gaussian.SpotVector import GaussSpot
from tools.gaussian.utils import real_to_range
from tools.gaussian.utils import RectangleDimentions, as_img_dims

class ShapeDetector(nn.Module):
    def __init__(
            self,
            out_size: int,
            image_size: RectangleDimentions,
            filters=16,
            pad="same",
            cut_below=0.03,
            device=torch.device("cpu"),
        ):
        super().__init__()
        out_size = out_size + 1 # +1 cell choice dimention
        self.device = device
        self.cut_below = cut_below
        self.layers = nn.Sequential(
            nn.Conv2d(1, filters, 3, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(filters),
            #nn.MaxPool2d(2),
            # maxpooling 2x2
            
            nn.Conv2d(filters, filters*2, 3, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(filters*2),
            nn.Conv2d(filters*2, filters*4, 3, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(filters*4),
            nn.MaxPool2d(2),
            # maxpooling 2x2
            
            nn.Conv2d(filters*4, filters * 4, 3, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(filters*4),
            nn.Conv2d(filters*4, filters * 8, 3, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(filters*8),
            #nn.Upsample(scale_factor=2),
            # maxpooling 2x2

            nn.Conv2d(filters * 8, filters * 8, 3, padding=pad),
            nn.BatchNorm2d(filters*8),
            nn.Conv2d(filters*8, out_size, 3, padding=pad),
        ).to(device)
        self._calc_xy_grid(image_size)

    def _calc_xy_grid(self, image_size: RectangleDimentions):
        SIZE = as_img_dims(image_size)
        output_size = self.calc_output_shape(1, 1, *SIZE)
        cell_size = ( SIZE[0] / output_size[-3], SIZE[1] / output_size[-2] )
        grid_shape = (SIZE[0]//cell_size[0], SIZE[1]// cell_size[1])
        X = torch.arange(0, grid_shape[0], device=self.device)*cell_size[0]
        Y = torch.arange(0, grid_shape[1], device=self.device)*cell_size[1]
        self.X, self.Y = torch.meshgrid(X, Y, indexing='ij')
        self.cell_size = cell_size
        self.image_size = SIZE

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = x.transpose(1, 2).transpose(2, 3) # B C W H --> B W H C(spot)

        center_choice = torch.sigmoid(x[...,-1])
        result = GaussSpot(x[...,:-1])
        return result, center_choice

    def forward_with_limits(self, x: torch.Tensor):
        if len(x.shape) == 3: #mabe assert instead?
            x = x.unsqueeze(1)
        result, center_choice = self.forward(x)
         #  limit values here
        result.intensity = torch.relu(result.intensity)   #common line
        result.muX = real_to_range(result.muX, 0, self.cell_size[0]) # do not use torus here!
        result.muY = real_to_range(result.muY, 0, self.cell_size[1])
        result.muX += self.X
        result.muY += self.Y 
        return result, center_choice

    def calc_output_shape(self, *input_shape: int):
        return self.forward(torch.randn(input_shape, device=self.device))[0].tensor.shape
    
    def _sum_image(self, gs: GaussSpot):
        return gs.sum_image(self.image_size, cut_below=self.cut_below , normalize=False, device=self.device)

    def _sum_images(self, result: GaussSpot, center_idxs: torch.Tensor):
        batch_size = result.tensor.shape[0]
        result_images = torch.zeros(batch_size, *self.image_size, device=self.device)
        for i in range(batch_size):
            result_images[i] = self._sum_image(result[i][center_idxs[i]])
        return result_images

    def forward_with_image(self, x: torch.Tensor):
        result, center_choice = self.forward_with_limits(x)
        images = self._sum_images(result, center_choice > 0.5)
        return result, center_choice, images

    # TODO: train on original data (with image loss as main loss)

    def train(self, 
        #batch_iterator: Iterable[Union[torch.Tensor, GaussSpot]], # TODO: Spot<> interface with certain method implemented (like draw, or )
        batch_iterator: Iterable[GaussSpot],
        lr=0.001,
        epochs=50,
        verbose: Union[int, bool]=10,
    ):    
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=lr
        )
        torch.set_grad_enabled(True)
        for ep in range(epochs):
            losses = []
            for iter_idx, batch in enumerate(batch_iterator):
                input_images = self._sum_image(batch)
                centers_idx = find_local_maximas(input_images, threshold=0., window=5)    
                # add random centers
                rnd_sparse = torch.rand(64, 64, device=self.device) > 0.99
                rand_empty = torch.logical_and(input_images == 0, rnd_sparse)
                centers_idx = torch.logical_or(centers_idx, rand_empty)
                
                batchMuX = batch.muX.unsqueeze(-1).unsqueeze(-1)
                batchMuY = batch.muY.unsqueeze(-1).unsqueeze(-1)  
                input_center_idx = torch.logical_and(
                    torch.logical_and(batchMuX > self.X, batchMuX <= self.X+self.cell_size[0]),
                    torch.logical_and(batchMuY > self.Y, batchMuY <= self.Y+self.cell_size[1]),
                ).sum(-3).to(torch.bool)
    
                result, center_choice = self.forward_with_limits(input_images)
                result_images = self._sum_images(result, input_center_idx)

                loss = F.mse_loss(result_images, input_images)
                loss += torch.abs(input_center_idx.to(torch.float) - center_choice).mean()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())  
                if (verbose if type(verbose) is bool else iter_idx%verbose == 0):
                    # fmt = "{:.6f}".format
                    # print("loss-10: ", sum(losses[-10:])/10)
                    print("Iter: ", iter_idx, " Loss: ", sum(losses)/len(losses))
                    losses = []
                
  
        torch.set_grad_enabled(False)
