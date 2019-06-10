# utils.py

import matplotlib  # For MACOSX
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torch.optim as optim

import time
import datetime
import math
import copy
from bisect import bisect
import os

####### metric for regularization in time
# regMetric = torch.dist # 2-norm
def regMetric(x,y):
    return torch.norm(x-y,p=1) # 1-norm
#######

dis = nn.CrossEntropyLoss()

def misfit(X,C):
    """
    softmax cross-entropy loss

    X - dims: nExamples -by- nFeats (+1)
    W - dims: nFeats +1 (if bias) -by- nClasses
    C - LongTensor containing class labels; dims = (nClasses by 1) where each entry is a value 0 to nClasses

    """

    # # remove the maximum for all examples to prevent overflow
    X = X.view(C.numel(), -1)
    S,tt = torch.max(X,1)
    S = X-S.view(-1,1)
    return dis(S,C), torch.exp(S)

def misfitW(X,W,C, device):
    """
    deprecated....but used in test functions currently
    softmax cross-entropy loss

    X - dims: nExamples -by- nFeats (+1)
    W - dims: nFeats +1 (if bias) -by- nClasses
    C - LongTensor containing class labels; dims = (nClasses by 1) where each entry is a value 0 to nClasses

    """

    featW = W.size(0)

    if X.size(1) + 1 == featW:
        # add bias
        e = torch.ones(X.size(0), 1, X.size(2), X.size(3)).to(device)
        X = torch.cat((X, e), dim=1)

    X = X.view(-1, featW)
    S = torch.matmul(X, W)
    # remove the maximum for all examples to prevent overflow
    Sm, tt = torch.max(S, 1)
    S = S - Sm.view(-1, 1)
    return dis(S, C), torch.exp(S)


def getAccuracy(S,labels):
    """compute accuracy"""
    _, predicted = torch.max(S.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct/total, correct, total

def conv1x1(x,K):
    """1x1 convolution with no padding"""
    return F.conv2d(x, K, stride=1, padding=0)

def conv3x3(x,K):
    """3x3 convolution with padding"""
    return F.conv2d(x, K, stride=1, padding=1)

def convt3x3(x,K):
    """3x3 convolution transpose with padding"""
    return F.conv_transpose2d(x, K, stride=1, padding=1)


def specifyGPU( gpuNumber ):
    """
    DEPRECATED
    directs operating system to only use the input gpu number
    :param gpuNumber: integer number associated with GPU you wish to use
                Entering a gpuNumber exceeding the number of GPUs you have, will result
                in the os choosing the cpu as the device
    """
    # import os

    nSystemGPUs = 2  #  we only have 2 GPUs on our server

    if gpuNumber > nSystemGPUs-1:
        print("Warning: specifyGPU: Do you have more than %d GPUs? If not, specifyGPU(%d) "
              "will have issues" % (nSystemGPUs, gpuNumber), file=sys.stderr)

    # restricts code to only seeing GPU labeled gpuNumber
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNumber)


def imshow(img):   # need to import matplotlib things
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def interpNN(theta,tTheta,tInter,inter=None, blankLayer=None, bKeepParam=False):
    """
    nearest neighbor interpolation
    TODO: refactor name .... this is really left neighbor interpolation
    """
    if len(theta) != len(tTheta):
        print('interLayer1D: lengths do not match')
        return -1

    nPoints = len(tInter)  # number of interpolation time points
    # make list of length same as tInter, and each element is of size K
    # assume all layers in the ModuleList theta have the same dimensions

    if inter is None:
        inter=nn.ModuleList()
        for i in range(nPoints):
            if blankLayer is None:
                module=copy.deepcopy(theta[0])
            else:
                module = copy.deepcopy(blankLayer)
                module.load_state_dict(theta[0].state_dict())
            inter.append(module)

    for k in range(nPoints):
        # get param for current time point
        # assume tTheta is sorted

        if tInter[k] <= tTheta[0]:   # if interp point is outside the tTheta range
            recursively_interpolate(inter[k], theta[0], theta[0], 1, 0, bKeepParam=bKeepParam)
        elif tInter[k] >= tTheta[-1]:
            recursively_interpolate(inter[k], theta[-1], theta[-1], 1, 0, bKeepParam=bKeepParam)
        else:
            idxTh = bisect(tTheta, tInter[k])
            # idxTh contains index of right point in tTheta to use for interpolation
            recursively_interpolate(inter[k], theta[idxTh-1], theta[idxTh-1], 1, 0, bKeepParam=bKeepParam)

    # ensure that running_mean and running_var send to gpu
    params = theta[0].parameters()
    first_param = next(params)
    device = first_param.device
    inter.to(device)
    return inter



def interpLayer1Dpreallocated(theta,tTheta,tInter,inter=None, blankLayer=None, bKeepParam=False):
    """
    1D (linear) interpolation. For the observations theta at tTheta, find the observations at points ti.
    - ASSUMPTIONS: all tensors in the list theta have the same dimension and tTheta is sorted ascending
    - theta are my K parameters
    :param theta:   list of LAYERS, think of them as measurements : f(x0) , f(x1), f(x2) , ...
    :param tTheta:  points where we have the measurements:  x0,x1,x2,...
    :param tInter:  points where we want to have an approximate function value: a,b,c,d,...
    :param blankLayer: module, an empty form of the layer to be interpolated (a workaround for when adding control layers)
    :param bKeepParam: boolean, maintain Parameter status (False will mean interpolated parameters will be reduced
                                to buffer status)
    :return: inter: approximate values f(a), f(b), f(c), f(d) using the assumption
                      that connecting every successive theta to its previous one with a line
    """
    if len(theta) != len(tTheta):
        print('interLayer1D: lengths do not match')
        return -1

    nPoints = len(tInter)  # number of interpolation time points
    # make list of length same as tInter, and each element is of size K
    # assume all layers in the ModuleList theta have the same dimensions

    if inter is None:
        inter=nn.ModuleList()
        for i in range(nPoints):
            if blankLayer is None:
                module=copy.deepcopy(theta[0])
            else:
                module = copy.deepcopy(blankLayer)
                module.load_state_dict(theta[0].state_dict())
            inter.append(module)


    for k in range(nPoints):
        # get K for current time point
        # assume tTheta is sorted

        if tInter[k] <= tTheta[0]:   # if interp point is outside the tTheta range
            recursively_interpolate(inter[k], theta[0], theta[0], 1, 0, bKeepParam=bKeepParam)
        elif tInter[k] >= tTheta[-1]:
            recursively_interpolate(inter[k], theta[-1], theta[-1], 1, 0, bKeepParam=bKeepParam)
        else:
            idxTh = bisect(tTheta, tInter[k])
            # idxTh contains index of right point in tTheta to use for interpolation
            leftTh  = tTheta[idxTh-1]
            rightTh = tTheta[idxTh]
            h       = rightTh - leftTh
            wtLeft  = (rightTh - tInter[k]) / h
            wtRight = (tInter[k] - leftTh) / h

            recursively_interpolate(inter[k], theta[idxTh - 1], theta[idxTh], wtLeft, wtRight, bKeepParam=bKeepParam)

    # ensure that running_mean and running_var send to gpu
    params = theta[0].parameters()
    first_param = next(params)
    device = first_param.device
    inter.to(device)
    return inter


def recursively_interpolate(output,left,right,wleft,wright, bKeepParam=False):
    """
        helper functions to interpolate the submodules and parameters
    :param output: module which holds interpolated values (a state layer)
    :param left:   module to the left, used for interpolation
    :param right:  module to the right, used for interpolation
    :param wleft:  float, weight for left item
    :param wright: float, weight for right item
    :param bKeepParam: boolean, maintain Parameter status (False will mean interpolated parameters will be reduced
                                to buffer status)
    :return:
    """
    # center, left right = c,l,r
    for (cChild,lChild,rChild) in zip(output.named_children(),left.named_children(),right.named_children()):
        # print(cChild[0],lChild[0],rChild[0])
        recursively_interpolate(cChild[1], lChild[1], rChild[1], wleft, wright, bKeepParam=bKeepParam)
    for (l, r) in zip(left.named_parameters(recurse=False), right.named_parameters(recurse=False)):
        # l[0] and r[0] hold names, l[1] and r[1] hold the values/Tensors
        if bKeepParam:
            exec( 'output.' + l[0] + '.data = wleft *l[1] + wright * r[1]') # find a better way to do this
        else: # replace parameter with a buffer
            output.__delattr__(l[0])
            output.register_buffer(l[0], wleft*l[1] + wright*r[1])







def listNorm(list1, list2):
    """
    RETIRED, used in tests
    input two lists of tensors of same length
    convert/vectorize the tensors in each list
    compute their 2-norm

    :param list1: [K1, K2, K3,....]  where Ki is a tensor
    :param list2: [A1, A2, A3,....]  where Ai is a tensor
    :return: float 2-norm
    """
    retValue = 0.0

    # use sqrt to iteratively update the norm for each element in the list
    for a,b in zip(list1,list2):
        a = a.contiguous().view(a.numel()) # vectorize
        b = b.contiguous().view(b.numel())  # vectorize
        retValue = torch.sqrt( retValue**2 + torch.norm(a-b,p=2)**2)

    return retValue




def getDataLoaders( sDataset , batchSizeTrain, device, batchSizeTest=None, percentVal = 0.20,
                    transformTrain=None, transformTest=None, nTrain=None, nVal=None, datapath=None):
    """
        set up data loaders
        following https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        for the training and validation split

    :param sDataset:       string, representation of the dataset, e.g. 'cifar10', 'cifar100', 'stl10'
    :param batchSizeTrain: int, batch size for the training data
    :param device:         pytorch device
    :param batchSizeTest:  int, batch size for the testing data
    :param percentVal:     float, percent of the training data to set aside as validation
    :param transformTrain: set of transforms for the training data
    :param transformTest:  set of transforms for the testing data
    :param nTrain:         int, number of training images   (None means use all, subject to percentVal)
    :param nVal:           int, number of validation images (None means use all, subject to percentVal)
     percentVal is applied first, then if nTrain or nVal is not None, then a smaller subset of the split data will be used
    :param datapath:       string, path to where the data is saved
    :return:
    """


    shift = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[1., 1., 1.],
    )

    # augmentations and convert to tensor
    if transformTrain is None:
        transformTrain = transforms.Compose(
            [transforms.ToTensor(),
             shift])
    if transformTest is None:
        transformTest = transforms.Compose(
            [transforms.ToTensor(),
             shift])

    if batchSizeTest is None:
        batchSizeTest = batchSizeTrain

    sDataPath= './data'
    if datapath is not None:
        sDataPath = datapath

    #----------------------------------------------------------------------------------
    #----------------------------------CIFAR-10----------------------------------------
    if sDataset == 'cifar10':

        trainSet = torchvision.datasets.CIFAR10(root=sDataPath, train=True,
                                                download=True, transform=transformTrain)
        valSet = torchvision.datasets.CIFAR10(root=sDataPath, train=True,
                                              download=True, transform=transformTest)
        testSet = torchvision.datasets.CIFAR10(root=sDataPath, train=False,
                                               download=True, transform=transformTest)
        nChanIn = 3
        nClasses = 10
    # ----------------------------------CIFAR-100----------------------------------------
    if sDataset == 'cifar100':
        trainSet = torchvision.datasets.CIFAR100(root=sDataPath, train=True,
                                                download=True, transform=transformTrain)
        valSet = torchvision.datasets.CIFAR100(root=sDataPath, train=True,
                                              download=True, transform=transformTest)
        testSet = torchvision.datasets.CIFAR100(root=sDataPath, train=False,
                                               download=True, transform=transformTest)
        nChanIn = 3
        nClasses = 100
    # ----------------------------------STL-10-----------------------------------------
    elif sDataset == 'stl10':

        trainSet = torchvision.datasets.STL10(root=sDataPath, split='train',
                                                download=True, transform=transformTrain)
        valSet = torchvision.datasets.STL10(root=sDataPath, split='train',
                                              download=True, transform=transformTest)
        testSet = torchvision.datasets.STL10(root=sDataPath, split='test',
                                               download=True, transform=transformTest)
        nChanIn = 3
        nClasses = 10
    # ----------------------------------------------------------------------------------

    # dimImg = trainSet.train_data.shape[1:3] # image dimensions

    nImages = len(trainSet)
    indices = list(range(nImages))
    nValImages = int(np.floor(percentVal * nImages))

    # np.random.seed(3)
    np.random.shuffle(indices)

    idxTrain, idxVal = indices[nValImages:], indices[:nValImages]

    if nTrain is not None:
        idxTrain = idxTrain[0:nTrain]
    if nVal is not None:
        idxVal = idxVal[0:nVal]

    samplerTrain = SubsetRandomSampler(idxTrain)
    samplerVal = SubsetRandomSampler(idxVal)

    loaderTrain = torch.utils.data.DataLoader(trainSet, batch_size=batchSizeTrain,
                                              sampler=samplerTrain)  # is pin_memory=True a good idea????????
    loaderVal = torch.utils.data.DataLoader(valSet, batch_size=batchSizeTest,
                                            sampler=samplerVal)
    loaderTest = torch.utils.data.DataLoader(testSet, batch_size=batchSizeTest,
                                             shuffle=False)

    print('training on %d images ; validating on %d images' % (len(idxTrain), len(idxVal)))

    return loaderTrain, loaderVal, loaderTest, nChanIn, nClasses



def printFeatureMatrix(nt, nBlocks, vFeat):
    """
    Construct and print a matrix that contains all of the channel sizes
    Example:
            Model architecture where each column is a layer
            no. Input Channels  [  3  32  32  32  32  64  64  64  64]
            no. Output Channels [ 32  32  32  32  64  64  64  64  64]
            conv kernel height  [  3   3   3   3   1   3   3   3   1]
            conv kernel width   [  3   3   3   3   1   3   3   3   1]
    :param nt:      number of time steps in each RK4 block
    :param nBlocks: number of blocks
    :param vFeat:   feature vector
    """
    mFeat = np.zeros((4, (nt + 1) * nBlocks + 1), dtype=int)

    k = 0
    # opening layer
    mFeat[0, k] = 3        # RGB input
    mFeat[1, k] = vFeat[0] # channels after opening layer
    mFeat[2, k] = 3        # 3x3 convolution
    mFeat[3, k] = 3
    k = k + 1

    # block and connecting layer
    for blk in range(nBlocks):
        for i in range(nt):
            mFeat[0, k] = vFeat[blk]
            mFeat[1, k] = vFeat[blk]
            mFeat[2, k] = 3
            mFeat[3, k] = 3
            k = k + 1

        mFeat[0, k] = vFeat[blk]
        mFeat[1, k] = vFeat[blk + 1]
        mFeat[2, k] = 1
        mFeat[3, k] = 1
        k = k + 1

    rowLabels = ["no. Input Channels ", "no. Output Channels", "conv kernel height ", "conv kernel width  "]
    print("\nModel architecture where each column is a layer")
    for rowLabel, row in zip(rowLabels, mFeat):
        print('%s [%s]' % (rowLabel, ' '.join('%03s' % i for i in row)))
    print("\n")
    return 0


def normalInit(dims):
    """
    Essentially, PyTorch's init.xavier_normal_ but clamped
    :param K: tensor to be initialized/overwritten
    :return:  initialized tensor on the device in the nn.Parameter wrapper
    """
    K = torch.zeros(dims)
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(K)
    sd = math.sqrt(2.0 / (fan_in + fan_out))
    # sd = math.sqrt(2 / (nChanLayer1 * 3 * 3 * 3)) # Meganet approach
    with torch.no_grad():
        K = K.normal_(0, sd)

    K = torch.clamp(K, min = -2*sd, max=2*sd)

    return  K


def appendTensorList(params, Klist, nWeights, weight_decay=None):
    """
    RETIRED. used in test functions
    helper function for adding a list of tensors to the params


    :param params: the params object that will be passed to the optimizer
    :param Klist:  list of Tensors that you want to be parameters for the optimizer
    :param nWeights: int, current number of weights in params (will be incremented)
    :param weight_decay: optional float, specific tikhinov regularization for Klist weights
        (specifically, if Klist are the scaling weights for a tvNorm, you may want no weight_decay)
    :return: updated params, updated number of weights in params
    """

    if weight_decay is not None:
        for K in Klist:
            nWeights += K.numel()
            params.append({'params': K , 'weight_decay': weight_decay})
    else:
        for K in Klist:
            nWeights += K.numel()
            params.append({'params': K})

    return params, nWeights






def getVectorizedParams(net,device):
    """
     obtain vectors of the running means, running variances, and parameters
     use built-in parameters_to_vector approach

    :param net:    the network
    :param device: retired/unused
    :return: vector of the running means, vector of the running variances, vector of the parameters
    """
    param_device = None

    vRunMean = []
    vRunVar = []

    for name, param in net.named_buffers():
        param_device = nn.utils.convert_parameters._check_param_device(param,param_device)
        if 'running_mean' in name:
            vRunMean.append(param.view(-1))
        elif 'running_var' in name:
            vRunVar.append(param.view(-1))

    runMean = torch.cat(vRunMean)
    runVar  = torch.cat(vRunVar)
    params  = nn.utils.convert_parameters.parameters_to_vector(net.parameters())

    return runMean, runVar, params


def trainNet(net,optimizer,lr,nEpoch, device, sBasePath, loaderTrain,loaderVal, nMini=1, verbose=True,
             regularization_param=0,checkpointing=False):
    """
        function for training the network

    :param net:         model to be trained
    :param optimizer:   pytorch optimizer
    :param lr:          numpy array of length nEpoch, learning for each epoch
    :param nEpoch:      int, number of epochs
    :param device:      pytorch device
    :param sBasePath:   string, path for results files (will append .txt , _model.pt , etc. for each file)
    :param loaderTrain: dataloader for training data
    :param loaderVal:   dataloader for validation sata
    :param nMini:       int, when on cpu/debugging, print out results for every nMini batches
    :param verbose:     boolean, when True and on cpu/debugging print out more often than every epoch
    :param regularization_param: float, weight applied to regularization metric (for regularization in time)
    :param checkpointing: boolean, if True use checkpointing
    :return:
    """

    # save the train-val split indices
    torch.save([loaderTrain.sampler.indices,loaderVal.sampler.indices], sBasePath + '_trainValSplit.pt')

    sPathOpt = ''
    valOpt = 100000  # keep track of optimal validation loss
    checkpt = time.time()

    results = np.zeros((nEpoch,10))

    oldRunMean, oldRunVar, oldParams = getVectorizedParams(net,device)


    regValue = 0
    if regularization_param>0:
        print('%-9s %-9s %-11s %-11s %-9s %-11s %-9s %-9s %-9s %-9s %-13s' %
              ('epoch', 'time', '|runMean|', '|runVar|',
               'lr', '|params|', 'avgLoss', 'acc', 'valLoss', 'valAcc', 'reg'), flush=True)

    else:
        print('%-9s %-9s %-11s %-11s %-9s %-11s %-9s %-9s %-9s %-9s' %
              ('epoch', 'time', '|runMean|', '|runVar|',
               'lr', '|params|', 'avgLoss', 'acc', 'valLoss', 'valAcc'), flush=True)


    for epoch in range(nEpoch):  # loop over the dataset multiple times

        """
        ########################## REMOVE THESE ################
        if epoch==100:
            if net.dynamicBlocks[0].controlLayers[0]._get_name() == 'DoubleLayer':
                removeBatchNormRKNet(net) # remove batch norm
                print('removing batch norm')
                oldRunMean, oldRunVar, oldParams = getVectorizedParams(net,device)
            else:
                regularization_param = 0 # for doubleSym remove regularization
        """

        # adjust learning rate by epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr.item(epoch)

        running_loss = 0.0
        running_num_correct = 0.0
        running_num_total   = 0.0
        running_reg  = 0.0
        for i, data in enumerate(loaderTrain, 0):

            net.train() # set model to train mode

            if checkpointing is True:
                loss, numCorrect, numTotal = net.checkpoint_train(data, optimizer, device,regularization_param)
            else:
            # regular optimization without checkpointing as before
            # get the inputs
                inputs, labels = data

            # zero the parameter gradients
                optimizer.zero_grad()

                inputs, labels = inputs.to(device), labels.to(device)
            # forward + backward + optimize

                x = net(inputs)
                loss, Si = misfit(x,labels)

                #add regularization to loss
                if regularization_param != 0:
                    regValue = regularization_param*net.regularization()
                    loss = loss + regValue
                loss.backward()
                optimizer.step()


                # impose bound constraints / clip the weights
                if hasattr(net, 'clip'): # some networks we created (ResNet and old RKNet) are not ClippedModules)
                    net.clip()


                _ , numCorrect, numTotal = getAccuracy(Si,labels)

            running_loss        += numTotal * loss.item()
            running_num_correct += numCorrect
            running_num_total   += numTotal
            running_reg += numTotal * regValue

            # print statistics
            # print every few mini-batches when on the CPU
            if device.type == 'cpu':
                if verbose == True:
                    if i % nMini == nMini - 1:  # print every nMini mini-batches
                        newRunMean, newRunVar, newParams = getVectorizedParams(net,device)
                        print('%-4d %-4d %-9.1f %-11.2e %-11.2e %-9.2e %-11.2e %-9.2e %-9.2f' %
                              (epoch + 1,
                               running_num_total,
                               time.time() - checkpt,
                               torch.norm(oldRunMean-newRunMean),
                               torch.norm(oldRunVar - newRunVar),
                               optimizer.param_groups[0]['lr'],
                               torch.norm(oldParams - newParams),
                               running_loss / running_num_total,
                               running_num_correct*100 / running_num_total),flush=True)

        ### EVALUATION
        # after 1 training epoch, validate

        valLoss, valAcc = evalNet(net, device, loaderVal)

        newRunMean, newRunVar, newParams = getVectorizedParams(net,device)

        results[epoch, 0] = epoch + 1
        results[epoch, 1] = time.time() - checkpt
        results[epoch, 2] = torch.norm(oldRunMean-newRunMean)
        results[epoch, 3] = torch.norm(oldRunVar - newRunVar)
        results[epoch, 4] = optimizer.param_groups[0]['lr']
        results[epoch, 5] = torch.norm(oldParams - newParams)
        results[epoch, 6] = running_loss / running_num_total
        results[epoch, 7] = running_num_correct * 100 / running_num_total
        results[epoch, 8] = valLoss
        results[epoch, 9] = valAcc * 100

        if regularization_param>0:
            print('%-9d %-9.1f %-11.2e %-11.2e %-9.2e %-11.2e %-9.2e %-9.2f %-9.2e %-9.2f %-13.2e' %
                  (results[epoch, 0],
                   results[epoch, 1],
                   results[epoch, 2],
                   results[epoch, 3],
                   results[epoch, 4],
                   results[epoch, 5],
                   results[epoch, 6],
                   results[epoch, 7],
                   results[epoch, 8],
                   results[epoch, 9],
                   running_reg / running_num_total),flush=True)
        else:
            print('%-9d %-9.1f %-11.2e %-11.2e %-9.2e %-11.2e %-9.2e %-9.2f %-9.2e %-9.2f' %
                  (results[epoch, 0],
                   results[epoch, 1],
                   results[epoch, 2],
                   results[epoch, 3],
                   results[epoch, 4],
                   results[epoch, 5],
                   results[epoch, 6],
                   results[epoch, 7],
                   results[epoch, 8],
                   results[epoch, 9]),flush=True)

        if valLoss < valOpt:
            # sPath = sBasePath + '_acc_' + "{:.3f}".format(valAcc)
            sPath = sBasePath + '_model'
            torch.save(net, sPath + '.pt') # save entire model
            valOpt = valLoss # update
            sPathOpt = sPath


        checkpt = time.time()



        oldRunMean, oldRunVar, oldParams = newRunMean, newRunVar, newParams

    # end for epoch

    saveResults(sBasePath + '_results.mat', results)

    return sPathOpt

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------




def evalNet(net, device, loaderTest):
    """
    evaluate the networks weights....
    use for both validation and test sets, just change loaderTest

    return loss, acc     (val_loss, val_acc  OR  test_loss, test_acc)
    """


    net.eval()  # set model to evaluation mode

    # compute validation accuracy
    vali = 0
    with torch.no_grad():
        # valRunLoss = 0.0
        weightedLoss = 0.0
        valRunCorrect = 0
        valRunTotal = 0

        for vali, valData in enumerate(loaderTest, 0):
            imagesVal, labelsVal = valData
            imagesVal, labelsVal = imagesVal.to(device), labelsVal.to(device)
            xVal = net(imagesVal)
            lossVal, SiVal = misfit(xVal, labelsVal)
            accVal, batchCorrect, batchCount = getAccuracy(SiVal, labelsVal)

            # valRunLoss    += lossVal.item()
            weightedLoss  += batchCount * lossVal.item()
            valRunCorrect += batchCorrect
            valRunTotal   += batchCount

            if device.type=='cpu': # for debugging
                break

    valAcc = valRunCorrect / valRunTotal
    valLoss = weightedLoss / valRunTotal     # valRunLoss / (vali + 1)

    return  valLoss , valAcc






#
# def helperDoubleSymRemoveBN(convLayer, normLayer):
#     """
#
#         FAILS TESTS BECAUSE of the activation function within
#
#         -K^T ( act( N ( K(Y) ) ) )
#         Easy to remove the N, but then no guarantee to maintain same weights for conv and conv transpose
#
#        incorporate the batch norm in the convolution
#
#        from -K^T ( s * (KY + beta + rm)/( sqrt( rv + eps ) ) + b ) + t
#
#        to    -K'^T ( K'*Y + beta' ) + t
#
#        where K' = s^(1/2) / (rv+eps)^(1/4) * K   and beta' =  (s^(1/2) / (rv+eps)^(1/4) )*(b + s*(beta - rm))
#
#        t    - convt bias
#        s    - norm scaling weight
#        b    - norm scaling bias
#        rm   - running_mean
#        rv   - running_variance
#        eps  - epsilon of norm
#        K    - convolution weights
#        beta - convolution bias
#
#        """
#
#     # convLayer  = layer.conv
#     # normLayer  = layer.normLayer
#     # transLayer = layer.convt
#
#     rm  = normLayer.running_mean.data
#     rv  = normLayer.running_var.data
#     s   = normLayer.weight.data
#     b   = normLayer.bias.data
#     epsil = normLayer.eps
#     beta  = convLayer.bias.data  # conv bias
#     K   = convLayer.weight.data
#
#
#     # new K and b
#
#     # denom = (rv+epsil )**(1/4) # remove the epsilon will give better results, but could fall victim to division by zero
#     denom = (rv) ** (1 / 4) + epsil**(9 / 8) # hack: epsilon tuned to give desired behavior
#
#     num = b + (s * (beta - rm))
#
#     sqs = torch.sqrt(s)
#     beta = (sqs / denom) * num
#
#     s = sqs/denom
#     s = s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#     K = s * K
#
#     convLayer.bias.data = beta
#     convLayer.weight.data = K




def helperRemoveBN(convLayer, normLayer):
    """ convLayer will absorb the information in normLayer assuming N( K(Y) )... norm of conv of features"""
    rm  = normLayer.running_mean.data
    rv  = normLayer.running_var.data
    s   = normLayer.weight.data
    b   = normLayer.bias.data
    epsil = normLayer.eps
    beta  = convLayer.bias.data  # conv bias
    K   = convLayer.weight.data

    # new K and b
    denom = torch.sqrt(rv + epsil)
    s = s / denom
    beta = b + (s * (beta - rm))
    s = s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    K = s * K

    convLayer.bias.data = beta
    convLayer.weight.data = K

    return 0


def removeBatchNorm(layer):
    """
    incorporate the batch norm in the convolution

    from s * (KY + beta + rm)/( sqrt( rv + eps ) ) + b

    to    K'*Y + beta'

    where K' = s / sqrt(rv+eps) * K   and beta' = b + s*(beta - rm)/sqrt(rv+eps)

    s    - norm scaling weight
    b    - norm scaling bias
    rm   - running_mean
    rv   - running_variance
    eps  - epsilon of norm
    K    - convolution weights
    beta - convolution bias

    """
    bUnchanged = 1

    if layer._get_name() == 'DoubleLayer' or layer._get_name() == 'PreactDoubleLayer':
        if hasattr(layer,'normLayer1'):
            if layer.normLayer1._get_name()=='BatchNorm2d':
                bUnchanged = helperRemoveBN(layer.conv1, layer.normLayer1)
                layer.__delattr__('normLayer1')
        if hasattr(layer, 'normLayer2'):
            if layer.normLayer2._get_name() == 'BatchNorm2d':
                bUnchanged = helperRemoveBN(layer.conv2, layer.normLayer2)
                layer.__delattr__('normLayer2')
    elif layer._get_name() == 'DoubleSymLayer':
        if layer.normLayer._get_name() == 'BatchNorm2d':
            print('no known way to remove batchnorm from DoubleSymLayer with a non-identity activation function')
    else:
        if hasattr(layer, 'normLayer'):
            if layer.normLayer._get_name() == 'BatchNorm2d':
                bUnchanged =helperRemoveBN(layer.conv, layer.normLayer)
                layer.__delattr__('normLayer')

    return bUnchanged

def removeBatchNormRKNet(net):
    """remove batch norm in the state layers and control layers

    Only works for rk1 , tY = tTheta


    for rk4 and rk1 with more states than layers....what do I do with the interpolation?
    This code will remove all the batch norms, but the resulting network will not be tuned
    to similar accuracy or performance

    """
    bUnchanged = 1
    #---------------------------------------------
    # Remove Batch Norm if exists and incorporate into the convolutions
    #---------------------------------------------

    # remove the batch norm
    for blk in net.dynamicBlocks: # for each dynamic block

        if blk._get_name() == 'rk1':
            states = blk.stateLayers
            nControlLayers = len(blk.controlLayers)-1
            reducedStates = states
        elif blk._get_name() == 'rk4': # account for the odd setup of the stateLayers in the rk4
            states = []
            reducedStates = []
            for i in range(len(blk.stateLayers)):
                reducedStates.append(blk.stateLayers[i][0])
                for j in [0,1,2,3]:
                    states.append(blk.stateLayers[i][j])

            reducedStates.append(blk.stateLayers[i][3]) # don't forget the last layer
            nControlLayers = len(blk.controlLayers)
        else:
            print('don\'t know what scheme is used in block')


        for l in states: # for each state layer in the dynamic block

            bUnchanged = removeBatchNorm(l)

        # need to remove it in the control layer as well
        if l._get_name() == 'DoubleLayer' or l._get_name() == 'PreactDoubleLayer':
            for i in range(nControlLayers):
                thisLayer = blk.controlLayers[i]
                if hasattr(thisLayer,'normLayer1'):
                    if thisLayer.normLayer1._get_name() == 'BatchNorm2d':
                        thisLayer.__delattr__('normLayer1')
                        thisLayer.conv1.weight.data = reducedStates[i].conv1.weight.data
                        thisLayer.conv1.bias.data   = reducedStates[i].conv1.bias.data
                if hasattr(thisLayer, 'normLayer2'):
                    if thisLayer.normLayer2._get_name() == 'BatchNorm2d':
                        thisLayer.__delattr__('normLayer2')
                        thisLayer.conv2.weight.data = reducedStates[i].conv2.weight.data
                        thisLayer.conv2.bias.data = reducedStates[i].conv2.bias.data

            if blk._get_name() == 'rk1':
                removeBatchNorm(blk.controlLayers[-1])


    if bUnchanged==1:
        print('No batch norms were removed')

    return net




# For all run functions
def getRunDefaults(sDataset, gpu=0):
    """
        the default running hyperparameters for experimentss
    :param sDataset: string, name of dataset, 'cifar10' , 'cifar100', 'stl10'
    :param gpu: int, None means use cpu; otherwise, signify which gpu to use
    :return: vFeat, transformTrain, transformTest, batchSize, lr, tTheta, nTrain, nVal, device
        vFeat - vector of number of features/channels per dynamic block
        transformTrain - default transformations for training dataloader
        transformTest  - default transformations for testing dataloader
        batchSize      - int, batch size
        lr             - numpy array, learning rate per epoch
        tTheta         - list, time discretization for controls
        nTrain         - int, number of training examples
        nVal           - int, number of validation examples
        device         - pytorch device
    """


    lr = np.concatenate((1e-1 * np.ones((80, 1), dtype=np.float32),
                         1e-2 * np.ones((40, 1), dtype=np.float32),
                         1e-3 * np.ones((40, 1), dtype=np.float32),
                         1e-4 * np.ones((40, 1), dtype=np.float32)), axis=0)


    tTheta=[0,1,2,3,4]

    if sDataset == 'cifar10' or sDataset=='cifar100':
        vFeat = [16,32,64,64]

        shift = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[1., 1., 1.],
        )

        # augmentations and convert to tensor

        transformTrain = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             shift])

        transformTest = transforms.Compose(
            [transforms.ToTensor(),
             shift])

        batchSize = 125

        if sDataset=='cifar10':
            # cifar10 is so much easier, 120 epochs is good enough
            lr = np.concatenate((1e-1 * np.ones((60, 1), dtype=np.float32),
                                 1e-2 * np.ones((20, 1), dtype=np.float32),
                                 1e-3 * np.ones((20, 1), dtype=np.float32),
                                 1e-4 * np.ones((20, 1), dtype=np.float32)), axis=0)

        if sDataset=='cifar100':
            lr = np.concatenate((1e-1 * np.ones((60), dtype=np.float32),
                                 1/np.geomspace(1/1e-1, 1/1e-2, num=20),
                                 1e-2 * np.ones((20), dtype=np.float32),
                                 1 / np.geomspace(1 / 1e-2, 1 / 1e-3, num=20),
                                 1e-3 * np.ones((20), dtype=np.float32),
                                 1 / np.geomspace(1 / 1e-3, 1 / 1e-4, num=20),
                                 1e-4 * np.ones((20), dtype=np.float32)), axis=0)
    elif sDataset == 'stl10':
        vFeat = [16, 32, 64, 128, 128]

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # augmentations and convert to tensor

        transformTrain = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
             transforms.RandomCrop(96, padding=12),
             transforms.ToTensor(),
             normalize])


        transformTest = transforms.Compose(
            [transforms.ToTensor(),
             normalize])

        batchSize = 30

    # TODO sDataset = tiny Image Net


    # if on CPU, run small version of the model
    if not torch.cuda.is_available() or gpu is None:
        print('running CPU default vFeat and lr')
        vFeat = [4, 7, 7]
        lr = np.array([1e-1, 1e-4])
        nTrain = batchSize*4 + 5
        nVal   = batchSize + 2
        device = torch.device('cpu')
    else:
        # specifyGPU(gpu)
        device = torch.device('cuda:' + str(gpu))
        nTrain = None
        nVal   = None

    return vFeat, transformTrain, transformTest, batchSize, lr, tTheta, nTrain, nVal, device












###################################################
# SAVING AND LOADING
###################################################

def saveResults(path, npArray):
    sio.savemat(path, {'results': npArray})

def loadResults(path):
    return sio.loadmat(path)['results']

def saveTensor(path, inputs):
    sio.savemat(path, {'inputs' : inputs.cpu().numpy()})

def loadTensor(path, inputs=None):
    loaded_inputs = torch.Tensor(sio.loadmat(path))
    if inputs is not None:
        inputs = loaded_inputs

    return loaded_inputs


def saveParams(path,Kopen,Kresnet,Kconnect,Knorm,KtvNorm,W, runMean, runVar):

    par = {}
    par["Kopen"] = Kopen.detach().cpu().numpy()
    for i in range(len(Kresnet)):
        key = "Kresnet_" + str(i)
        par[key] = Kresnet[i].detach().cpu().numpy()
    for i in range(len(Kconnect)):
        key = "Kconnect_" + str(i)
        par[key] = Kconnect[i].detach().cpu().numpy()
    for i in range(len(Knorm)):
        key = "Knorm_" + str(i)
        par[key] = Knorm[i].detach().cpu().numpy()
    for i in range(len(KtvNorm)):
        key = "KtvNorm_" + str(i)
        par[key] = KtvNorm[i].detach().cpu().numpy()
    par["W"] = W.detach().cpu().numpy()

    for i in range(len(runMean)):
        key = "runMean_" + str(i)
        par[key] = runMean[i].detach().cpu().numpy()
    for i in range(len(runVar)):
        key = "runVar_" + str(i)
        par[key] = runVar[i].detach().cpu().numpy()

    # print(par)
    # data = {'parameters': par}
    sio.savemat(path, par)
    return 0

def loadParams(path,Kopen,Kresnet,Kconnect,Knorm,KtvNorm,W,runMean, runVar, device):
    loaded_params = sio.loadmat(path)

    # loaded_params['W'] - W.detach().numpy()   # should be all 0s
    # torch.Tensor(loaded_params['W']) - W      # should be all 0s

    # load them back into the parameters
    for name in loaded_params.keys():
        if name[0] != '_':  # ignore the titles with first char '_'    ,      __headers__ and such
            if '_' in name:
                word, idx = str.split(name, '_')
                if word == 'Kconnect':
                    vars()[word][int(idx)].data = torch.Tensor(loaded_params[name]).to(device)
                else:
                    vars()[word][int(idx)].data = torch.Tensor(loaded_params[name].squeeze()).to(device)
            else:
                vars()[name].data = torch.Tensor(loaded_params[name].squeeze()).to(device)

    return loaded_params


def testSaveLoad(Kopen,Kresnet,Kconnect,Knorm,KtvNorm,W):
    import copy

    # import h5py    # import tables      # pytables
    # save model
    # PATH = 'modelWeights.idk'
    # torch.save(KtvNorm, PATH)



    oldParams = copy.deepcopy([Kopen] + Kresnet + Kconnect + Knorm + KtvNorm + [W])

    saveParams('par1.mat', Kopen, Kresnet, Kconnect, Knorm, KtvNorm, W)

    # clear   # use zero_  ???
    # Kopen.data = torch.zeros(Kopen.shape)
    Kopen.data.zero_()
    for i in range(len(Kresnet)):
        Kresnet[i].data = torch.zeros(Kresnet[i].shape)
    for i in range(len(Kconnect)):
        Kconnect[i].data = torch.zeros(Kconnect[i].shape)
    for i in range(len(Knorm)):
        Knorm[i].data = torch.zeros(Knorm[i].shape)
    for i in range(len(KtvNorm)):
        KtvNorm[i].data = torch.zeros(KtvNorm[i].shape)
    W.data = torch.zeros(W.shape)

    loadParams('par1.mat',Kopen,Kresnet,Kconnect,Knorm,KtvNorm,W)

    print("\nnorm difference")
    print(listNorm(oldParams, [Kopen] + Kresnet + Kconnect + Knorm + KtvNorm + [W]))
    print('\n')





# Testing functions



def func(x,K,K2):
    # z = 2*K
    z = conv3x3(x,K)
    # z = DoubleSymLayer(x,K)
    # z = tvNorm(K)
    # z = rk4DoubleSymBlock(x, [K,K2], tvnorm=False, weightBias = [None]*10)

    # vFeat = [4,8,12,16]
    # nChanIn = 3
    # nClasses = 10
    # tY = [0,1,2,3,4]
    # tTheta = tY
    # dynamicScheme = rk4
    # layer = DoubleSymLayer
    # net = RKNet(tY, tTheta, nChanIn, nClasses, vFeat, dynamicScheme=dynamicScheme, layer=layer)
    #

    return z

def dCheckDoubleSym():
    # derivative check for the DoubleSym Layer
    # define function f(b)
    # b = torch.Tensor([1.,1.,2.,2.])
    b  = torch.randn(4,4,3,3)
    b2 = torch.randn(4,4,3,3)
    b.requires_grad = True

    x = torch.randn(2, 4, 3, 3)  # 2 images, 4 channels, 3x3 pixels
    x.requires_grad = False
    # x = torch.randn(4)

    # f = DoubleSymLayer(x, b, tvnorm=False)
    # f = conv3x3(x,b)
    # f =  2 * b

    f = func(x, b, b2)

    v = torch.randn(4*4*3*3)
    # v = torch.randn(4)

    # K = torch.Tensor([5,7,11,13])
    # K = torch.Tensor(torch.randn(4, 4, 3, 3))
    K = torch.randn(4,4,3,3)
    K2 = torch.randn(4,4,3,3)
    dx = torch.ones(2, 4, 3, 3)

    torch.autograd.backward([f], [dx, K, K2] ) # stores gradient of f with respect to K in b.grad
    J = b.grad.view(1,-1)

    Jv = torch.matmul(J,v)


    err = np.zeros((3,30))
    for k in range(1,30):
        h = 2**(-k)

        # fK = 2*K
        fK = func(x,K, K2)

        first = func(x, K+(h*v).view(K.shape), K2) - fK

        # first = DoubleSymLayer( x, K+(h*v).view(K.shape) , tvnorm=False ) - f
        # first = conv3x3(x, K + (h * v).view(K.shape)) - f
        # first = 2* ( K + (h * v).view(K.shape) ) - fK

        E0 = torch.norm(first)
        E1 = torch.norm(first - h*Jv)

        print('h=%1.2e  E0=%1.2e  E1=%1.2e' % (h,E0,E1))
        err[0, k - 1] = h
        err[1, k - 1] = E0
        err[2, k - 1] = E1


    return err





def inter1D(theta,tTheta,tInter):
    """
    1D (linear) interpolation. For the observations theta at tTheta, find the observations at points ti.
    - ASSUMPTIONS: all tensors in the list theta have the same dimension and tTheta is sorted ascending
    - theta are my K parameters
    :param theta:   list of Tensors, think of them as measurements : f(x0) , f(x1), f(x2) , ...
    :param tTheta:  points where we have the measurements:  x0,x1,x2,...
    :param tInter:  points where we want to have an approximate function value: a,b,c,d,...
    :return: inter: approximate values f(a), f(b), f(c), f(d) using the assumption
                      that connecting every successive theta to its previous one with a line
    """
    if len(theta) != len(tTheta):
        print('inter1D: lengths do not match')
        return -1

    nPoints = len(tInter)  # number of interpolation time points
    # make list of length same as tInter, and each element is of size K
    # interpK = thi
    # assume all tensors in the list theta have the same dimensions
    inter = [torch.zeros(theta[0].shape)] * nPoints


    for k in range(nPoints):
        # get K for current time point
        # assume tTheta is sorted
        if tInter[k] <= tTheta[0]:   # if interp point is outside the tTheta range
            inter[k] = 1*theta[0]
        elif tInter[k] >= tTheta[-1]:
            inter[k] = 1*theta[-1]
        # elif tInter[k] in tTheta:  # if interp point is already in the list, give that function value
        #     idxTh = tTheta.index(tInter[k])
        #     inter[k] = theta[idxTh]
        else:
            idxTh = bisect(tTheta, tInter[k])
            # idxTh contains index of right point in tTheta to use for interpolation
            leftTh   = tTheta[idxTh-1]
            rightTh  = tTheta[idxTh]
            h      = rightTh - leftTh
            wtLeft  = (rightTh - tInter[k]) / h
            wtRight = (tInter[k] - leftTh) / h
            inter[k] = wtLeft*theta[idxTh-1] + wtRight*theta[idxTh]

    return inter



def testInter1D():
    """test example for inter1D"""
    t = torch.ones(2,1)
    K = [2*t, 4*t,6*t,5*t]
    tTheta = [1,2,4,5]
    ti = [0.7,1.0,1.9,2.0,2.1,2.4,2.7,3,3.3,5.0, 5.4]
    interpolatedK = inter1D(K, tTheta, ti)
    print(interpolatedK)
    return 0


