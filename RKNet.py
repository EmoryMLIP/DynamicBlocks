# RKNet.py

from utils import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import time
import datetime
import math
import copy

import os
import sys

from modules.ConnectingLayer import ConnectingLayer
from modules.rk4 import rk4
from modules.rk1 import rk1
from modules.TvNorm import TvNorm
from modules.DoubleSymLayer import DoubleSymLayer
from modules.DoubleLayer import DoubleLayer

from modules.ClippedModule import *

class RKNet(ClippedModule):
    def __init__(self, tY, tTheta, nChanIn, nClasses, vFeat, dynamicScheme=rk4,  layer=DoubleSymLayer,
                 openParams=None, dynamicParams=None, connParams=None, linear=None):
        """
            The main workhorse network. Consists of an opening layer (a single layer), multiple dynamicBlocks each
            of which is connected by a ConnectingLayer (a single layer), then

        :param tY:       list, time discretization for the states , [0,1,2,3,4]
        :param tTheta:   list, time sdiscretization for the controls, [0,1,2,3,4]
        :param nChanIn:  int,  number of input channels (RGB = 3)
        :param nClasses: int,  number of output classes
        :param vFeat:    list, number of channels per dynamic block...last value for final connecting layer [16,32,64,64]
        :param dynamicScheme: module, the time integrator / ODE solver (ex: rk1, rk4)
        :param layer:         module, the primary layer in the dynamic blocks (ex: DoubleLayer, DoubleSymLayer)
        :param openParams:    dict, the params dict to pass to the opening layer
        :param dynamicParams: dict, the params dict to pass to the dynamic scheme
                also, can pass strings "TvNorm", "NoNorm", or "Batch" for the corresponding norm with the default
        :param connParams:    dict, the params dict to pass to the connecting layers
        :param linear:        module, a nn.Linear module already initialized to be used as final fully connected layer
        """
        super().__init__()
        self.nBlocks = len(vFeat) - 1
        self.vFeat  = vFeat
        self.tY     = tY
        self.tTheta = tTheta

        if dynamicParams is None or dynamicParams=="TvNorm" or dynamicParams=="Batch" or dynamicParams=="NoNorm":
            dynamicParams = [dynamicParams] * self.nBlocks
        if connParams is None:
            connParams = [None] * self.nBlocks


        self.dynamicBlocks = nn.ModuleList([])
        self.connectors    = nn.ModuleList([])

        act = nn.ReLU()

        if openParams is None:
            openParams = {  'act': act,
                            'normLayer': nn.BatchNorm2d(num_features=vFeat[0], eps=1e-4),
                            'conv': nn.Conv2d(nChanIn, vFeat[0], kernel_size=3, padding=1, stride=1)}
        self.open = ConnectingLayer([nChanIn, vFeat[0]], params=openParams)


        self.dynamicBlocks = nn.ModuleList([])
        self.connectors    = nn.ModuleList([])

        for blk in range(self.nBlocks):

            if dynamicParams[blk] is None or dynamicParams[blk]=="TvNorm":
                # default setup for the dynamic block
                dynamicParams[blk] = {'act': act,
                                      'vFeat' : vFeat[blk],
                                      'normLayer': TvNorm(vFeat[blk], eps=1e-4)}
            elif dynamicParams[blk]=="Batch":
                dynamicParams[blk] = {'act': act,
                                      'vFeat': vFeat[blk],
                                      'normLayer': nn.BatchNorm2d(vFeat[blk])}
            elif dynamicParams[blk]=="NoNorm":
                dynamicParams[blk] = {'act': act,
                                      'vFeat': vFeat[blk]}


            self.dynamicBlocks.append(dynamicScheme(tTheta, tY, layer, layerParams = dynamicParams[blk] ))

            if connParams[blk] is None:
                connParams[blk] = {'act': act,
                                   'normLayer': nn.BatchNorm2d(num_features=vFeat[blk+1], eps=1e-4),
                                   'conv': nn.Conv2d(vFeat[blk], vFeat[blk+1], kernel_size=1, padding=0, stride=1)}

            connLayer = ConnectingLayer( [vFeat[blk],vFeat[blk+1]] , params = connParams[blk])
            self.connectors.append(connLayer)

        if linear is None:
            self.linear = nn.Linear(vFeat[-1], nClasses)
            self.linear.weight.data = normalInit(self.linear.weight.shape)
            self.linear.bias.data = self.linear.bias.data*0
        else:
            self.linear = linear


    def forward(self, x):

        x = self.open(x)

        for blk in range(self.nBlocks):

            x = self.dynamicBlocks[blk](x)
            x = self.connectors[blk](x)

            if blk < self.nBlocks - 1:
                # x = self.connectors[blk](x) # move connectors here ?????????????? Kind of a big change????????
                x = F.avg_pool2d(x, 2) # TODO: make the pooling operator abstract for max vs avg pooling etc.
            else:
                # average each channel
                x = F.avg_pool2d(x, x.shape[2:4])

        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


    def setup_checkpointing(self,optimizer):

        self.optimizers=[]

        opt1=copy.deepcopy(optimizer)
        opt1.defaults=optimizer.defaults
        opt1.param_groups=[]

        opt1.add_param_group({"params": self.open.parameters()})
        self.optimizers.append(opt1)
        for blk in range(self.nBlocks):
            opt = copy.deepcopy(optimizer)
            opt.defaults = optimizer.defaults
            opt.param_groups = []

            opt.add_param_group({"params":self.dynamicBlocks[blk].parameters()})
            opt.add_param_group({"params":self.connectors[blk].parameters()})
            self.optimizers.append(opt)
        opt = copy.deepcopy(optimizer)
        opt.defaults = optimizer.defaults
        opt.param_groups = []

        opt.add_param_group({'params': self.linear.weight})
        self.optimizers.append(opt)

    def checkpoint_train(self,data,optimizer,device,regularization=0):
        self.setup_checkpointing(optimizer)
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        y=[]
        inputs, labels = data
        self.train() # set model to train mode
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            # forward + backward + optimize

            x = self.open(inputs)
            y.append(x)        # checkpointing
            for blk in range(self.nBlocks):
                x = self.dynamicBlocks[blk](x)
                x = self.connectors[blk](x)
                if blk < self.nBlocks - 1:
                    x = F.avg_pool2d(x, 2)
                else:
                    x = F.avg_pool2d(x, x.shape[2:4])

            # x = x.view(x.size(0), -1)
            # x = self.linear(x)

                y.append(x)  # checkpointing

        with torch.enable_grad():
            y_tag=Variable(y[-1],requires_grad=True)

            # net.linear.weight.data = torch.transpose(net.W[0:-1, :], 0, 1)
            # net.linear.bias.data = net.W[-1, :]


            loss, Si = misfitW(y_tag, torch.transpose(self.linear.weight,0,1), labels, device)
            final_loss=loss.item()


            loss.backward()
            prev_grad = y_tag.grad

            ## check what we have on hand
            self.optimizers[-1].step()
            self.optimizers[-1].zero_grad()
            for blk in reversed(range(self.nBlocks)):
                y_tag = Variable(y[-2], requires_grad=True)
                x = self.dynamicBlocks[blk](y_tag)
                x = self.connectors[blk](x)
                if blk < self.nBlocks - 1:
                    x = F.avg_pool2d(x, 2)  # TODO: make the pooling operator abstract for max vs avg pooling etc.
                else:
                    # average each channel
                    x = F.avg_pool2d(x, x.shape[2:4])
                loss = torch.dot(x.view(-1),prev_grad.view(-1))
                loss=loss+regularization*sum(self.dynamicBlocks[blk].weight_variance())
                y.pop()
                loss.backward()
                prev_grad = y_tag.grad

                self.optimizers[1+blk].step()
                self.optimizers[1+blk].zero_grad()


        x = self.open(inputs)
        loss = torch.dot(x.view(-1),prev_grad.view(-1))
        y.pop()
        loss.backward()
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()

        _ , numCorrect, numTotal = getAccuracy(Si,labels)
        torch.cuda.empty_cache()
        return final_loss , numCorrect, numTotal



    def checkpoint_train_debug(self,data,optimizer,device,net_other):
        net_other.train()  # set model to train mode

        inputs, labels = data
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        z = net_other.forward(inputs)
        aa=[]
        aa.append(inputs)
        x = self.open(inputs)
        aa.append(x)
        for blk in range(self.nBlocks):

            x = self.dynamicBlocks[blk](x)
            x = self.connectors[blk](x)

            if blk < self.nBlocks - 1:
                x = F.avg_pool2d(x, 2)  # TODO: make the pooling operator abstract for max vs avg pooling etc.
            else:
                # average each channel
                x = F.avg_pool2d(x, x.shape[2:4])
            aa.append(x)


        loss_other, Si = misfit(z, net_other.linear.weight, labels, device)

        loss_other.backward()


        self.setup_checkpointing(optimizer)
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        y=[]
        inputs, labels = data
        self.train() # set model to train mode
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            # forward + backward + optimize
            y.append(inputs)  # checkpointing
            x = self.open(inputs)
            y.append(x)        # checkpointing
            for blk in range(self.nBlocks):
                x = self.dynamicBlocks[blk](x)
                x = self.connectors[blk](x)
                if blk < self.nBlocks - 1:
                    x = F.avg_pool2d(x, 2)
                else:
                    x = F.avg_pool2d(x, x.shape[2:4])

                y.append(x)  # checkpointing

        with torch.enable_grad():
            y[-1]=Variable(y[-1],requires_grad=True)


            for reg,chkpt in zip(aa,y):
                print("forward vs checkpoint:")
                print(torch.norm(reg - chkpt))
                print(reg.shape)


            loss, Si = misfit(y[-1], self.linear.weight, labels, device)
            final_loss=loss.item()


            print("loss is:")
            print(torch.norm(loss-loss_other))


            loss.backward()


            print("linear.weight:")
            print(torch.norm(self.linear.weight-net_other.linear.weight))
            print(torch.norm(self.linear.weight.grad-net_other.linear.weight.grad))




            ## check what we have on hand
            self.optimizers[-1].step()
            self.optimizers[-1].zero_grad()
            for blk in reversed(range(self.nBlocks)):
                y[-2] = Variable(y[-2], requires_grad=True)
                x = self.dynamicBlocks[blk](y[-2])
                x = self.connectors[blk](x)
                if blk < self.nBlocks - 1:
                    x = F.avg_pool2d(x, 2)  # TODO: make the pooling operator abstract for max vs avg pooling etc.
                else:
                    # average each channel
                    x = F.avg_pool2d(x, x.shape[2:4])
                loss = torch.dot(x.view(-1),y[-1].grad.view(-1))
                loss.backward()
                y.pop()

                print("connectors " + str(blk)+ ":")
                P1=self.connectors[blk].parameters()
                P2=net_other.connectors[blk].parameters()
                for p1,p2 in zip(P1,P2):
                    print(torch.norm(p1.grad-p2.grad)/torch.norm(p1.grad))
                    print(p1.shape)
                print("dynamicBlocks " + str(blk)+ ":")
                P1=self.dynamicBlocks[blk].parameters()
                P2=net_other.dynamicBlocks[blk].parameters()
                for p1,p2 in zip(P1,P2):
                    print(torch.norm(p1.grad-p2.grad)/torch.norm(p1.grad))
                    print(p1.shape)

                self.optimizers[1+blk].step()
                self.optimizers[1+blk].zero_grad()


        x = self.open(y[0])
        loss = torch.dot(x.view(-1),y[1].grad.view(-1))
        y.pop()
        loss.backward()
        print("Opening :")
        P1 = self.open.parameters()
        P2 = net_other.open.parameters()
        for p1, p2 in zip(P1, P2):
            print(torch.norm(p1.grad - p2.grad)/torch.norm(p1.grad))
        print("finished")
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()
        y.pop()
        _ , numCorrect, numTotal = getAccuracy(Si,labels)
        torch.cuda.empty_cache()
        return final_loss , numCorrect, numTotal


    def regularization(self):
        reg=0
        for block in self.dynamicBlocks:
            reg = reg + sum(block.weight_variance())
        #print(reg)
        return reg


def runRKNet(sDataset, sTitle, net=None,
             tTheta=None, tY=None, vFeat = None,
             dynamicScheme=rk4, dynamicParams=None, layer=DoubleSymLayer,
             writeFile=True, gpu=0, fTikh=4e-4 , fMomentum=0.9, bNesterov=False,
             transTrain=None, transTest=None, lr=None, batchSize=None,
             regularization_param=0.0,checkpointing=False,reparametrization_epochs=None,
             nTrain=None, nVal=None, percentVal=0.2,
             loaderTrain=None, loaderVal=None, loaderTest=None, nChanIn=3, nClasses=None):
    """

    :param sDataset:   string, name of dataset ; 'cifar10' , 'stl10' , 'cifar100'
    :param sTitle:     string, prefix to name output files
    :param net:        Module, the network if want different from the default network
    :param tTheta:     list  , time discretization for the controls (where the weights are)
    :param tY:         list  , time discretization for the states   (where the layers are)
    :param vFeat:      list  , the width/ number of channels for each dynamic block (last entry for last connecting block)
    :param dynamicScheme: Module name, time integrator/ ODE solver (ex: rk1, rk4)
    :param layer:      Module name, layer used for dynamic block (ex: DoubleLayer, DoubleSymLayer)
    :param writeFile:  boolean, True means write the log to a text file, False means print to terminal
    :param gpu:        int  , which gpu to use to operate (must be < number of gpus accessible)
    :param fTikh:      float, weight decay / tikhonov regularization scalar for optimizer
    :param fMomentum:  float, momentum value for optimizer
    :param bNesterov:  boolean, True means use Nesterov, False means no Nesterov for optimizer
    :param lr:         numpy array, learning rate per epoch (length=number of epochs)
    :param batchSize:  int, number of input examples (images) per batch
    :param transTrain: pytorch transform, defines the augmentation and conversion to Tensor for training set
    :param transTest:  pytorch transform, defines the augmentation and conversion to Tensor for testing/validation set
    :param regularization_param: float, for the regularization in time within each dynamic block, the loss
                becomes: Loss + regularization_param * regMetric(prevLayer,currLayer)   (regMetric defined in utils)
    :param checkpointing: boolean, True means implement checkpointing, False runs normally
    :param reparametrization_epochs:

        To use entire dataset, leave nTrain and nVal as None. nTrain and nVal values override percentVal.
    :param nTrain:     int, number of training examples (must be <= total examples in training set)
    :param nVal:       int, number of validation examples ( must be <= total examples in validation set)
    :param percentVal: float, percentage of training set that will be split out as validation

        if passing one of the following, must pass all 5:
            loaderTrain, loaderVal, loaderTest, nChanIn, and nClasses
    :param loaderTrain: dataloader for training data
    :param loaderVal:   dataloader for validation data
    :param loaderTest:  dataloader for testing data
    :param nChanIn:     int, number of input channels
    :param nClasses:    int, number of output classes
    :return: net, sBasePath, device
        net:       the model
        sBasePath: string, path used to name files; (ex. sBasePath_model.pt, sBasePath.txt)
        device:    torch device used to train the model
    """


    start = time.time() # timing

    sStartTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print('start time: ', sStartTime)
    sBasePath = 'results/' + sTitle + '_' + sDataset + '_' + sStartTime
    print('file: ', sBasePath)


    # get defaults -----------------------------------------------------------------------------
    vFeatDef, transTrainDef, transTestDef, batchSizeDef, lrDef, tThetaDef, nTrainDef, nValDef, device = \
                                                                                getRunDefaults(sDataset, gpu=gpu)

    if writeFile: # write output to a text file instead of stdout
        sys.stdout = open(sBasePath + '.txt' , 'wt' )


    # For the option to reparametrize during training:
    if reparametrization_epochs is not None:
        tY_array=tY
        tTheta_array=tTheta
        tY=tY[0]
        tTheta=tTheta[0]
    else:
        tY_array = None
        tTheta_array = None

    if net is None:
        if vFeat is None:
            vFeat = vFeatDef
        if tTheta is None:
            tTheta = tThetaDef
    else:
        vFeat = net.vFeat
        tTheta = net.tTheta
        tY = net.tY

    if transTrain is None:
        transTrain = transTrainDef
    if transTest is None:
        transTest = transTestDef
    if batchSize is None:
        batchSize = batchSizeDef
    if lr is None:
        lr = lrDef
    if nTrain is None:
        nTrain = nTrainDef
    if nVal is None:
        nVal = nValDef
    if tY is None:
        tY = tTheta


    # percentVal = 0.20 # use 20% of the training set as validation
    # fTikh      = 4e-4 # tikhonov regularization / weight decay value
    # fMomentum  = 0.9
    # bNesterov  = False

    #-------------------------------------------------------------------------------------------

    nEpoch = lr.size
    print("device", device)

    print('time steps Y:      ' , tY)
    print('time steps theta:  ' , tTheta)
    print('no. of channels:   ' , vFeat)
    print('no. of epochs:     ' , nEpoch)
    print('ODE reg param:     ' , regularization_param)
    print('Use checkpointing: ' , checkpointing)


    # assume that if trainLoader is passed, then all loaders are passed
    if loaderTrain is None:
        if nTrain is not None and nVal is not None:
            loaderTrain, loaderVal, loaderTest, nChanIn, nClasses = \
                getDataLoaders(sDataset, batchSize, device,
                               batchSizeTest=batchSize, percentVal=percentVal,
                               transformTrain=transTrain, transformTest=transTest,
                               nTrain=nTrain, nVal=nVal)
        else:
            loaderTrain, loaderVal, loaderTest, nChanIn, nClasses = \
                    getDataLoaders( sDataset , batchSize, device,
                                    batchSizeTest=batchSize, percentVal=percentVal,
                                    transformTrain=transTrain, transformTest=transTest,
                                    nTrain=None, nVal=None )





    # print the matrix presenting the channels and conv sizes for every layer
    # printFeatureMatrix(nt, nBlocks, vFeat)

    if net is None:
        net = RKNet(tY, tTheta, nChanIn, nClasses, vFeat, dynamicScheme=dynamicScheme, dynamicParams=dynamicParams, layer=layer )

    net.to(device)

    # count number of weights and layers
    nWeights = sum(p.numel() for p in net.parameters())
    nLayers  = len(list(net.parameters()))
    print('Training ',nWeights,' weights in ', nLayers, ' layers',flush=True)

    # separate out the norms
    normParams = []
    convParams = []
    for name, param in net.named_parameters():
        if 'normLayer' in name:
            normParams.append(param)
        else:
            convParams.append(param)

    allParams = [ {'params': normParams, 'weight_decay': 0 },
                  {'params': convParams} ]

    optimizer = optim.SGD( allParams, lr=lr.item(0), momentum=fMomentum, weight_decay=fTikh, nesterov=bNesterov)
    print('optimizer = SGD with momentum=%.1e, weight_decay=%.1e, nesterov=%d, batchSize=%d'
            % (fMomentum, fTikh, bNesterov, batchSize))

    # train the network
    sPathOpt = trainNet(net, optimizer, lr, nEpoch, device, sBasePath, loaderTrain, loaderVal, nMini=1, verbose=True,
                        regularization_param=regularization_param,checkpointing=checkpointing)

    end = time.time()
    print('Time elapsed: ' , end-start)
    print('Training complete. Now testing...')

    #--------------TESTING-----------------
    # evaluate performance on test set

    # load best model weights based on validation accuracy
    net = torch.load(sPathOpt + '.pt')
    testLoss, testAcc = evalNet(net, device, loaderTest)

    print('\ntesting loss: %-9.2e    testing accuracy: %-9.2f' %
            ( testLoss,  testAcc * 100   ) )

    print(net) # just so we can see what does what

    return net, sBasePath, device

    print('done')




def runDefault():
    runRKNet( sDataset='cifar10', sTitle = 'default_RK4_DoubleSym', writeFile=False)

def runDefaultRegularized():
    runRKNet( sDataset='stl10', sTitle = 'regularizedBN_RK1_Double', dynamicScheme=rk1, layer = DoubleLayer,
            writeFile=True,regularization_param=0.01, checkpointing=False, gpu=1, dynamicParams=["Batch"])


    # For Testing!!
if __name__ == '__main__':
    runDefault()
    # runDefaultRegularized()


