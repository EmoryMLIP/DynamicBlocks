# rk1.py

import torch
import torch.nn as nn
from utils import interpLayer1Dpreallocated
import copy

from modules.ClippedModule import *

# TODO: clean up

class rk1(ClippedModule):
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
            self.controlLayers = nn.ModuleList([layer(vFeat, copy.deepcopy(layerParams)) for i in range(self.nTheta)])
            # nn.ModuleList works like a python list of Modules

            # the hi and intermediate Ys for interpolation
            self.hY = [next - curr for curr, next in zip(self.tY, self.tY[1:])]
            self.midY = [curr + h / 2 for curr, h in zip(self.tY, self.hY)]
            # self.stateLayers = []
            self.stateLayers = self.interpolate(self.controlLayers, self.tTheta, self.tY[0: -1])


    def setTimeSteps(self, tY, tTheta, blankLayer=None, lastUnused=False, interpolate=None):
        """
        have the capability to set tY and tTheta between forward passes

        :param tY:  new state layer discretization
        :param tTheta: new control layer discretization
        :param blankLayer: supply a blank control layer for when new tTheta differs from prev tTheta
        :param lastUnused: pass True when the last layer of tTheta is unused (ex. when prev tTheta= prev tY=[0,1,2,3,4])
            in these cases, we want to exclude this untuned layer bc it could negatively influence the interpolation
        """

        # TODO: give ability to vary boundary conditions
        # hard-coded boundary conditions are to maintain no change from nearest knot

        if interpolate is None:
            if hasattr(self, 'interpolate'):
                interpolate = self.interpolate
            else: # patch for old networks
                self.interpolate = interpLayer1Dpreallocated


        if self.tTheta != tTheta:
            if blankLayer is None:
                print('must provide blankLayer if changing tTheta') # TODO: maybe can make more robust:
                                                                    # check if lengths are the same
                print('OR wait for torch to fix: \'Only Tensors created explicitly by the user support deepcopy\'')

            if lastUnused: # last control Layer is not used in RK1
                self.controlLayers = interpolate(self.controlLayers[0:-1], self.tTheta[0:-1], tTheta,
                                                               blankLayer=blankLayer, bKeepParam=True)
            else:
                self.controlLayers = interpolate(self.controlLayers, self.tTheta, tTheta, blankLayer=blankLayer,
                                                        bKeepParam = True)

        self.tY = tY
        self.tTheta = tTheta
        # the hi and intermediate Ys for interpolation
        self.hY = [next - curr for curr, next in zip(self.tY, self.tY[1:])]
        self.midY = [curr + h / 2 for curr, h in zip(self.tY, self.hY)]


        if lastUnused:
            self.stateLayers = interpolate(self.controlLayers[0:-1], self.tTheta[0:-1], self.tY[0: -1],blankLayer=blankLayer)

        else: # for rk1 don't use the last point
            self.stateLayers = interpolate(self.controlLayers, self.tTheta, self.tY[0: -1],blankLayer=blankLayer)

    def forward(self, x):
        if not hasattr(self, 'interpolate'): # patch for old networks
            self.interpolate = interpLayer1Dpreallocated

        nY = len(self.tY)

        self.stateLayers = self.interpolate(self.controlLayers, self.tTheta, self.tY[0: -1], self.stateLayers)

        for k in range(nY-1):
            # interpolate the fixed control layers (the Parameters) at tTheta locations to get the
            # interpolated state layers at the tY locations

            hi = self.hY[k]

            z1 = self.stateLayers[k](x)

            x = x + hi * z1

        return x

    def train(self, mode=True):
        self.training = mode
        nY = len(self.tY)

        for k in range(nY - 1):
            for module in self.stateLayers:
                module.train(mode)

        for module in self.controlLayers:
            module.train(mode)

        return self


    def weight_variance(self):
        weight_diffs=[]
        nTheta = len(self.tTheta)
        for k in range(nTheta - 1):
            fReg = self.controlLayers[k].weight_variance(self.controlLayers[k+1]).unsqueeze(0)
            weight_diffs.append( self.hY[k] * fReg )  # h * ||x-y||_p
        return weight_diffs

    def double_time_steps(self):
        ty=self.tY
        nY = len(self.tY)
        newtY=[]
        for i in range(nY-1):
            newtY.append(ty[i])
            newtY.append((ty[i]+ty[i+1])/2)
        newtY.append(ty[-1])
        self.setTimeSteps(newtY,self.tTheta)



if __name__ == "__main__":
    print('rk1 block test\n')
    import testRK1Block  # this should run the test

