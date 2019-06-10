# DoubleSymLayer.py

from utils import *
import torch.nn as nn
import copy
from modules.ClippedModule import *

class DoubleSymLayer(ClippedModule):

    def __init__(self, vFeat, params={}):
        super().__init__()
        if type(vFeat) is not list: # assume its one number
            vFeat = [vFeat, vFeat]
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
            szKernel  = self.conv.kernel_size[0]   # doesn't allow for rectangular kernels
            stride    = self.conv.stride
            padding   = self.conv.padding

        if 'szKernel' in params.keys():
            szKernel  = params.get('szKernel')

        if 'act' in params.keys():
            self.act  = params.get('act')

        if 'normLayer' in params.keys():
            self.normLayer = copy.deepcopy(params.get('normLayer'))
            self.normLayer.weight.data = torch.ones(nChanOut)
            #self.normLayer.bias.data = torch.zeros(nChanOut) # this may be redundant

        self.convt = nn.ConvTranspose2d(in_channels=nChanOut, kernel_size=szKernel,
                                        out_channels=nChanIn, stride=stride, padding=padding)

        # conv and convt need to share weights
        self.weight = nn.Parameter( normalInit([vFeat[1], vFeat[0], szKernel, szKernel]) , requires_grad=True)
        self.conv.weight = self.weight
        self.convt.weight = self.weight

        # initiate the biases to 0
        if self.conv.bias is not None:
            self.conv.bias.data  *= 0
        if self.convt.bias is not None:
            self.convt.bias.data *= 0

    def forward(self,x):
        z = self.conv(x)
        if hasattr(self, 'normLayer'):
            z = self.normLayer(z)
        z = self.act(z)
        z = - self.convt(z)
        return z

    def calcClipValues(self,h,nPixels,nChan):
        # DoubleSym should have bound constraints half of those in DoubleLayer
        super().calcClipValues(h,nPixels, nChan)
        # self.minDef  = 0.5*self.minDef
        # self.maxDef  = 0.5*self.maxDef
        self.minConv = 0.5*self.minConv
        self.maxConv = 0.5*self.maxConv

    # def weight_variance(self,other):
    #     value=0
    #     value+= torch.dist(self.weight, other.weight)
    #     return value


    def weight_variance(self,other):
        value = 0
        value += regMetric( nn.utils.convert_parameters.parameters_to_vector(self.parameters()) ,
                            nn.utils.convert_parameters.parameters_to_vector(other.parameters()) )
        return value

if __name__ == "__main__":
    print('DoubleSymLayer test\n')
    import testDoubleSymLayer  # this should run the test
