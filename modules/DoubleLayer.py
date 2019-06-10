# DoubleLayer.py
# 2/5/19

from utils import *
import torch.nn as nn
import copy
from modules.ClippedModule import *


class DoubleLayer(ClippedModule):
    """
    Implementation of the double layer, also referred to as a Basic ResNet Block.

    Attributes:
        conv1 (sub-module): convolution class, default is 3x3 2Dconvolution
        conv2 (sub-module):           ''
        act1  (sub-module): activation function, default is ReLU()
        act2  (sub-module):           ''
        normLayer1 (sub-module): normalization with affine bias and weight, default is no normalization
        normLayer2 (sub-module):      ''

    Typical attributes for the children:
        conv#.weight (Parameter):  dims (nChanOut,nChanIn,3,3) for default 2DConvolution from nChanIn -> nChanOut channels
        conv#.bias   (Parameter):  vector, dims (nChanIn)
        normLayer#.weight (Parameter): vector, dims (nChanOut) affine scaling
        normLayer#.bias   (Parameter): vector, dims (nChanOut) affine scaling bias
    """

    def __init__(self, vFeat, params={}):
        """
        :param vFeat: 2-item list of number of expected input channels and number of channels to return, [nChanIn,nChanOut]
        :param params: dict of possible parameters ( 'conv1' , 'conv2', 'act1' , 'act2' , 'normLayer1' , 'normLayer2' )
        """
        super().__init__()
        if type(vFeat) is not list: # assume its one number
            vFeat = [vFeat, vFeat]
        nChanIn = vFeat[0]
        nChanOut = vFeat[1]

        # defaults
        szKernel = 3
        stride   = 1
        padding  = 1
        # be cognisant of where you initialize...make sure each is initialized or deep-copied
        self.conv1 = nn.Conv2d(in_channels=nChanIn, kernel_size=szKernel,
                              out_channels=nChanOut, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=nChanIn, kernel_size=szKernel,
                               out_channels=nChanOut, stride=stride, padding=padding)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        # overwrite from params where necessary
        if 'conv1' in params.keys():
            self.conv1 = copy.deepcopy(params.get('conv1'))
        if 'conv2' in params.keys():
            self.conv2 = copy.deepcopy(params.get('conv2'))

        if 'act1' in params.keys():
            self.act1  = params.get('act1')
        if 'act2' in params.keys():
            self.act2 = params.get('act2')

        if 'normLayer1' in params.keys():
            self.normLayer1 = copy.deepcopy(params.get('normLayer1'))
            self.normLayer1.weight.data = torch.ones(nChanOut)
            self.normLayer1.bias.data   = torch.zeros(nChanOut) # this may be redundant
        if 'normLayer2' in params.keys():
            self.normLayer2 = copy.deepcopy(params.get('normLayer2'))
            self.normLayer2.weight.data = torch.ones(nChanOut)
            self.normLayer2.bias.data   = torch.zeros(nChanOut) # this may be redundant

        # assume if blah is passed instead of blah1 and blah2, then the user wants them the same
        if 'conv' in params.keys():
            self.conv1 = copy.deepcopy(params.get('conv'))
            self.conv2 = copy.deepcopy(self.conv1)
        if 'act' in params.keys():
            self.act1  = params.get('act')
            self.act2  = copy.deepcopy(self.act1)
        if 'normLayer' in params.keys():
            self.normLayer1 = copy.deepcopy(params.get('normLayer'))
            self.normLayer1.weight.data = torch.ones(nChanOut)
            #self.normLayer1.bias.data = torch.zeros(nChanOut)
            self.normLayer2 = copy.deepcopy(self.normLayer1)

        self.conv1.weight.data = normalInit(self.conv1.weight.data.shape)
        self.conv2.weight.data = normalInit(self.conv2.weight.data.shape)
        # for this demonstration, make sure the biases are 0
        if self.conv1.bias is not None:
            self.conv1.bias.data  *= 0
        if self.conv2.bias is not None:
            self.conv2.bias.data *= 0


    def forward(self,x):
        z = self.conv1(x)
        if hasattr(self, 'normLayer1'):
            z = self.normLayer1(z)
        z = self.act1(z)

        z = self.conv2(z)
        if hasattr(self, 'normLayer2'):
            z = self.normLayer2(z)
        z = self.act2(z)

        return z

    # def weight_variance(self,other):
    #     value=0
    #     value+= torch.dist(self.conv1.weight, other.conv1.weight)
    #     value += torch.dist(self.conv2.weight, other.conv2.weight)
    #     return value

    def weight_variance(self,other):
        value = 0
        value += regMetric( nn.utils.convert_parameters.parameters_to_vector(self.parameters()) ,
                            nn.utils.convert_parameters.parameters_to_vector(other.parameters()) )
        return value

if __name__ == "__main__":
    print('DoubleLayer test\n')
    import testDoubleLayer  # this should run the test
