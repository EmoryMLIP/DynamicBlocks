# rk4.py

import torch
import torch.nn as nn
from utils import interpLayer1Dpreallocated
from modules.ClippedModule import *

class rk4(ClippedModule):
    """
    Implementation of the Runge-Kutta 4 Dynamic Block

    Attributes:
        interpolate: function, the type of interpolation to perform  (default is 1D)
        tTheta:      list, time discretization of the controls
        tY:          list, time discretization of the states
        nTheta:      int, number of control layers
        hY:          list, the step sizes between consecutive time discretization points of tY
        midY:        list, the midpoints between subsequent tY
        controlLayers (sub-module list): all elements are instances of layer representing the control layers, with associated weights
        stateLayers   (sub-module list): all elements are instances of layer representing the state layers, with no Parameters
    """
    def __init__(self, tTheta, tY, layer, layerParams={}):
        super().__init__()

        self.interpolate = interpLayer1Dpreallocated

        if 'vFeat' in layerParams.keys():
            vFeat = layerParams.get('vFeat')
            del layerParams['vFeat']
        else:
            print('rk4: Error: layerParams must have \'vFeat\' declared')
            exit(1)


        self.tTheta = tTheta
        self.tY = tY
        self.nTheta = len(tTheta)

        # set up control layers....these need to contain the nn.Parameters and will be used for
        # the interpolation of the state layers
        self.controlLayers = nn.ModuleList([layer(vFeat, layerParams) for i in range(self.nTheta)])
        # nn.ModuleList works like a python list of Modules

        # the hi and intermediate Ys for interpolation
        self.hY = [next - curr for curr, next in zip(self.tY, self.tY[1:])]
        self.midY = [curr + h / 2 for curr, h in zip(self.tY, self.hY)]
        self.stateLayers=[]
        for k in range(len(self.tY) - 1):
            self.stateLayers.append( self.interpolate(self.controlLayers, self.tTheta,
                                            [self.tY[k],self.midY[k],self.midY[k],self.tY[k+1]]))

    def setTimeSteps(self, tY, tTheta, blankLayer=None, interpolate=None):
        """
        have the capability to set tY and tTheta between forward passes

        :param tY:     list, time discretizations for the states
        :param tTheta: list, time discretizations for the controls
        :param blankLayer:  module, blueprint control layer needed to be passed when adding control layers
        :param interpolate: function, interpolation method (default is 1D interpolation)
        """

        if interpolate is None:
            if hasattr(self, 'interpolate'):
                interpolate = self.interpolate
            else: # patch for old networks
                self.interpolate = interpLayer1Dpreallocated

        self.controlLayers =interpolate(self.controlLayers, self.tTheta,tTheta, blankLayer=blankLayer, bKeepParam=True)

        self.tY = tY
        self.tTheta = tTheta
        # the hi and intermediate Ys for interpolation
        self.hY = [next - curr for curr, next in zip(self.tY, self.tY[1:])]
        self.midY = [curr + h / 2 for curr, h in zip(self.tY, self.hY)]

        # if rediscretizing with a batch norm, the interpolation will set running_mean and variance to 0 and 1
        self.stateLayers = []
        for k in range(len(self.tY) - 1):
            self.stateLayers.append(interpolate(self.controlLayers, self.tTheta,
                                        [self.tY[k], self.midY[k], self.midY[k], self.tY[k + 1]],blankLayer=blankLayer))

    def forward(self, x):
        if not hasattr(self, 'interpolate'): # patch for old networks
            self.interpolate = interpLayer1Dpreallocated

        nY = len(self.tY)
        # interpolate the fixed control layers (the Parameters) at tTheta locations to get the
        # interpolated state layers at the tY locations

        for k in range(nY - 1):
            # adjust the weights that will be used for the layers....curr, mid, next
            self.interpolate(self.controlLayers, self.tTheta,
                        [self.tY[k], self.midY[k], self.midY[k], self.tY[k + 1]],self.stateLayers[k])

            # INSIGHT FROM PLOTTING REGULARIZED MODEL: the runMean and runVar, of both midY are similar,
            # as are those for k+1 and then the next k


            hi = self.hY[k]

            # first intermediate step
            z1 = self.stateLayers[k][0](x)

            # second intermediate step
            z2 = self.stateLayers[k][1](x + z1 * (hi / 2))

            # third intermediate step
            z3 = self.stateLayers[k][2](x + z2 * (hi / 2))

            # fourth intermediate
            z4 = self.stateLayers[k][3](x + z3 * hi)

            x = x + (hi/6) * (z1 + 2 * z2 + 2 * z3 + z4)

        return x

    def train(self, mode=True):
        """set train/eval mode for state layers b/c they're not in a ModuleList"""
        self.training = mode
        nY = len(self.tY)

        for k in range(nY - 1):
            for module in self.stateLayers[k]:
                module.train(mode)

        for module in self.controlLayers:
            module.train(mode)

        return self

    def weight_variance(self):
        """apply regularization in time"""
        weight_diffs=[]
        nTheta = len(self.tTheta)
        for k in range(nTheta - 1):
            fReg = self.controlLayers[k].weight_variance(self.controlLayers[k+1]).unsqueeze(0)
            weight_diffs.append( self.hY[k] * fReg )  # h * ||x-y||_p
        return torch.cat(weight_diffs)





if __name__ == "__main__":
    print('rk4 block test\n')
    import testRK4Block  # this should run the test

