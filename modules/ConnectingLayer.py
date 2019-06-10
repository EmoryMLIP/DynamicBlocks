# ConnectingLayer.py

import torch
import torch.nn as nn
from utils import normalInit, misfitW, conv3x3
import torch.optim as optim
import torch.nn.functional as F
import copy

from modules.ClippedModule import *


class ConnectingLayer(ClippedModule):
    """
    Implementation of the resizing connecting layer (changing the number of channels)

    Attributes:
        conv (sub-module): convolution class, default is 3x3 2Dconvolution
        act  (sub-module): activation function, defualt is ReLU()
        normLayer (sub-module): normalization with affine bias and weight, default is no normalization

    Typical attributes for the children:
        conv.weight (Parameter):  dims (nChanOut,nChanIn,3,3) for default 2DConvolution from nChanIn -> nChanOut channels
        conv.bias   (Parameter):  vector, dims (nChanIn)
        normLayer.weight (Parameter): vector, dims (nChanOut) affine scaling
        normLayer.bias   (Parameter): vector, dims (nChanOut) affine scaling bias

    """
    def __init__(self, vFeat, params={}):
        """
        :param vFeat: 2-item list of number of expected channels and number of channels to return, [nChanIn,nChanOut]
        :param params: dict of possible parameters ( 'conv' , 'act', 'szKernel' , 'normLayer' )
        """
        super().__init__()
        nChanIn = vFeat[0]
        nChanOut = vFeat[1]

        # defaults
        szKernel = 3
        stride   = 1
        padding  = 1
        self.conv = nn.Conv2d(in_channels=nChanIn, kernel_size=szKernel,
                              out_channels=nChanOut, stride=stride, padding=padding)
        self.act = nn.ReLU()

        # overwrite from params where necessary
        if 'conv' in params.keys():
            self.conv = copy.deepcopy(params.get('conv'))
            szKernel  = self.conv.kernel_size[0]  # doesn't allow for rectangular kernels
            stride    = self.conv.stride
            padding   = self.conv.padding
        elif 'szKernel' in params.keys():
            szKernel = params.get('szKernel')
            self.conv = nn.Conv2d(in_channels=nChanIn, kernel_size=szKernel,
                                  out_channels=nChanOut, stride=stride, padding=padding)

        if 'act' in params.keys():
            self.act = params.get('act')

        if 'normLayer' in params.keys():
            self.normLayer = copy.deepcopy(params.get('normLayer'))
            self.normLayer.weight.data = torch.ones(nChanOut)
            # self.normLayer.bias.data   = torch.zeros(nChanOut) # this may be redundant


        self.conv.weight.data = normalInit(self.conv.weight.data.shape)
        # make sure the biases are 0
        if self.conv.bias is not None:
            self.conv.bias.data *= 0



    def forward(self, x):
        z = self.conv(x)
        if hasattr(self, 'normLayer'):
            z = self.normLayer(z)
        z = self.act(z)
        return z



if __name__ == "__main__":
    print('ConnectingLayer test\n')
    import testConnectingLayer  # this should run the test





