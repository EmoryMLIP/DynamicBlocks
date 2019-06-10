# PreactDoubleLayer.py
# 2/5/19

from utils import *
import torch.nn as nn
import copy

from modules.DoubleLayer import DoubleLayer


class PreactDoubleLayer(DoubleLayer):
    """ pre-activated version of the DoubleLayer

        N2( act2( K2( N1( act1( K1(Y) )))))
    """

    def __init__(self, vFeat, params={}):
        super().__init__(vFeat, params=params)

    def forward(self,x):
        z = self.act1(x)
        z = self.conv1(z)
        if hasattr(self, 'normLayer1'):
            z = self.normLayer1(z)

        z = self.act2(z)
        z = self.conv2(z)
        if hasattr(self, 'normLayer2'):
            z = self.normLayer2(z)

        return z


if __name__ == "__main__":
    print('PreactDoubleLayer test\n')
    import testPreactDoubleLayer  # this should run the test
