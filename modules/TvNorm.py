# TvNorm.py

import torch
import torch.nn as nn

class TvNorm(nn.Module):
    """
    normalization using the total variation; idea is to normalize pixel-wise by the length of the feature vector, i.e.,
        MATLAB notation:
        z = diag( 1/ sqrt( sum(x.^2,3)+eps)) x

    Attributes:
        eps:    small float so no division by 0
        weight: scaling weight for the affine transformation
        bias:   bias for the affine transformation

    """
    def __init__(self, nChan, eps=1e-4):
        """
        :param nChan: number of channels for the data you expect to normalize
        :param eps: small float so no division by 0
        """
        super().__init__()

        self.eps  = eps

        # Tv Norm has no tuning of the scaling weights
        # self.weight = nn.Parameter(torch.ones(nChan))
        self.register_buffer('weight', torch.ones(nChan))

        self.bias   = nn.Parameter(torch.zeros(nChan))


    def forward(self,x):
        """
        :param x: inputs tensor, second dim is channels
                  example dims: (num images in the batch , num channels, height , width)
        :return: normalized version with same dimensions as x
        """
        z = torch.pow(x, 2)
        z = torch.div(x, torch.sqrt(torch.sum(z, dim=1, keepdim=True) + self.eps))
        # assumes that Tensor is formatted (  something , no. of channels, something, something, etc.)

        if self.weight is not None:
            w = self.weight.unsqueeze(0)  # add first dimension
            w = w.unsqueeze(-1)  # add last dimension
            w = w.unsqueeze(-1)  # add last dimension
            z = z * w
        if self.bias is not None:
            b = self.bias.unsqueeze(0)  # add first dimension
            b = b.unsqueeze(-1)  # add last dimension
            b = b.unsqueeze(-1)  # add last dimension
            z = z + b

        return z

