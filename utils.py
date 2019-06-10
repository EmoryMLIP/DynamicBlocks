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

####### metric for regularization
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

    # featW = W.size(0)
    #
    # if X.size(1) + 1 == featW:
    #     # add bias
    #     e = torch.ones(X.size(0),1,X.size(2),X.size(3)).to(device)
    #     X = torch.cat((X, e), dim=1)
    #
    # X = X.view(-1,featW)
    # S = torch.matmul(X,W)
    # # remove the maximum for all examples to prevent overflow
    X = X.view(C.numel(), -1)
    S,tt = torch.max(X,1)
    S = X-S.view(-1,1)
    return dis(S,C), torch.exp(S)

def misfitW(X,W,C, device):
    """
    Deprecated....but used in test functions currently
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


# newVector
def initVec(c,device,stdv = None,dist='normal'):
    # b = nn.Parameter(torch.Tensor(np.asscalar(c), device=device))
    b = nn.Parameter(torch.Tensor(c, device=device))

    if stdv is None:
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(b)
        stdv = math.sqrt(2.0 / (fan_in + fan_out))

    if dist == 'normal':
        b.data.normal_(0.0, stdv)
    elif dist == 'uniform':
        b.data.uniform_(0.0, stdv)
    b.data = b.data.to(device)
    return b




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



# TODO: do NOT interpolate activation functions
# TODO: interpolate the named_buffers so running mean and running variance are interpolated
# TODO: is recursive necessary? what if used paramters_to_vector ??? would that be more efficient?
def recursively_interpolate(output,left,right,wleft,wright, bKeepParam=False):
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





######################################################
# pre-training functions
######################################################

def getDataLoaders( sDataset , batchSizeTrain, device, batchSizeTest=None, percentVal = 0.20,
                    transformTrain=None, transformTest=None, nTrain=None, nVal=None, datapath=None):
    # set up data loaders
    # following https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
    # for the training and validation split

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





# TODO: refactor to getNetworkGeometry
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




def xavierInitialization(K,device):
    """
    Essentially, PyTorch's init.xavier_normal_ but clamped
    :param K: tensor to be initialized/overwritten
    :return:  initialized tensor on the device in the nn.Parameter wrapper
    """
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(K)
    sd = math.sqrt(2.0 / (fan_in + fan_out))
    # sd = math.sqrt(2 / (nChanLayer1 * 3 * 3 * 3)) # Meganet approach
    with torch.no_grad():
        K = K.normal_(0, sd)

    K = torch.clamp(K, min = -2*sd, max=2*sd)

    return nn.Parameter( K.to(device), requires_grad=True)



def initParams(nChanIn, nClasses,vFeat,lenTheta,device):
    """
    initialize all of the RK4 weights

    :param nChanIn:   number of channels into the network
    :param nClasses:  number of predicition classes
    :param vFeat:     vector of feature sizes / channel counts for the network geometry
    :param lenTheta:  number of theta time steps ( length of tTheta )
    :param device:    torch.device (usually 'cpu' or 'cuda')
    :return:  Kopen,Kresnet,Kconnect,Knorm,KtvNorm,W
    """


    nChanOut = vFeat[-1]        # number of output channels at end of model
    # nt = len(tY)              # number of time steps in each block
    # lenTheta = len(tTheta)
    nBlocks = len(vFeat) - 1    # number of rk4 blocks
    nChanLayer1 = vFeat[0]      # number of channels after opening layer

    ### Initialize Weights
    Knorm   = []

    normWeight = nn.Parameter( torch.ones(nChanLayer1).to(device) , requires_grad=True)
    normBias   = nn.Parameter( torch.zeros(nChanLayer1).to(device), requires_grad=True)
    Knorm.append(normWeight)
    Knorm.append(normBias)

    # first layer
    Kopen = xavierInitialization( torch.zeros(nChanLayer1,nChanIn,3,3), device )

    # resnet and connecting layers
    Kresnet = []
    KtvNorm = []
    Kconnect = []
    for blk in range(nBlocks):
        nChan = vFeat[blk]
        # connecting layers
        K = xavierInitialization( torch.zeros(vFeat[blk+1], vFeat[blk], 1, 1 ), device )
        Kconnect.append(K)

        # normalization in connecting layers
        normWeight = nn.Parameter(torch.ones(vFeat[blk+1]).to(device), requires_grad=True)
        normBias   = nn.Parameter(torch.zeros(vFeat[blk+1]).to(device), requires_grad=True)
        Knorm.append(normWeight)
        Knorm.append(normBias)

        # rk4 layers
        K = xavierInitialization(torch.zeros(nChan, nChan, 3, 3), device)
        for k in range(lenTheta):
            K = nn.Parameter(K.clone(), requires_grad=True)
            Kresnet.append(K)

        # tv Normalization in DoubleSym Layer
        normWeight = nn.Parameter(torch.ones(vFeat[blk]).to(device), requires_grad=True)
        normBias   = nn.Parameter(torch.zeros(vFeat[blk]).to(device), requires_grad=True)
        for l in range(lenTheta):
            normWeight = nn.Parameter(normWeight.clone(), requires_grad=True)
            normBias = nn.Parameter(normBias.clone(), requires_grad=True)
            KtvNorm.append(normWeight)
            KtvNorm.append(normBias)

    # classifier weights
    stdv = 0.1
    W = stdv*torch.randn(nChanOut+1,nClasses)             # plus one for the bias
    W = torch.clamp(W,min = -2*stdv, max = 2*stdv)
    W = nn.Parameter(W.to(device), requires_grad=True)

    return Kopen,Kresnet,Kconnect,Knorm,KtvNorm,W



def appendTensorList(params, Klist, nWeights, weight_decay=None):
    """
    helper function for adding a list of tensors to the params (formParamStruct)

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


def formParamStruct(Kopen, Kresnet, Kconnect, Knorm, KtvNorm, W):


    ######### DONT DO THIS:::: { 'params': list of tensors }
    # params = [{'params':Kopen   },
    #           {'params':Kresnet },
    #           {'params':Kconnect},
    #           {'params':Knorm   , 'weight_decay':0},
    #           {'params':KtvNorm , 'weight_decay':0},
    #           {'params':W      }]

    params = []
    # count up the number of parameters
    nWeights = Kopen.numel()
    params.append({'params': Kopen})
    params, nWeights = appendTensorList(params, Kresnet, nWeights)
    params, nWeights = appendTensorList(params, Kconnect, nWeights)
    # overrride defined weight_decay so no regularization exists on the normalization parameters
    params, nWeights = appendTensorList(params, Knorm, nWeights, weight_decay=0)
    # KtvNorm[1::2] is just the bias terms....we do NOT include the scaling weight because we want them
    # to remain fixed at 1
    params, nWeights = appendTensorList(params, KtvNorm[1::2], nWeights, weight_decay=0)
    nWeights += W.numel()
    params.append({'params': W})

    return params, nWeights









##########################################
# performing the optimization
##########################################

# def getVectorizedParams(net,device):
#     vRunMean = torch.Tensor().to(device)
#     vRunVar = torch.Tensor().to(device)
#     vParams = torch.Tensor().to(device)
#
#     for name, param in net.named_buffers():
#         if 'running_mean' in name:
#             vRunMean = torch.cat((vRunMean, param), dim=0)
#         elif 'running_var' in name:
#             vRunVar = torch.cat((vRunVar, param), dim=0)
#     for param in net.parameters():
#         vParams = torch.cat((vParams, param.view(1, -1).squeeze()), dim=0)
#
#     runMean = copy.copy(vRunMean)
#     runVar = copy.copy(vRunVar)
#     params = copy.copy(vParams)
#
#     return runMean, runVar, params

def getVectorizedParams(net,device):
    # use built-in parameters_to_vector approach
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

    # save the train-val split indices
    torch.save([loaderTrain.sampler.indices,loaderVal.sampler.indices], sBasePath + '_trainValSplit.pt')

    sPathOpt = ''
    valOpt = 100000  # keep track of optimal validation loss
    checkpt = time.time()

    results = np.zeros((nEpoch,10))

    oldRunMean, oldRunVar, oldParams = getVectorizedParams(net,device)
    # boundConstraints = clipWeights()
    # convConstraints  = clipConvs()

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





#### INADVISABLE TO USE UNLESS YOU KNOW WHAT YOU"RE DOING
# randomly adjust the time steps at each epoch
def trainNetRandom(net,optimizer,lr,nEpoch, device, sBasePath, loaderTrain,loaderVal,
                   nMini=1, verbose=True, tYnoise=0.2, tThetaNoise=0.2):

    sPathOpt = ''
    valOpt = 0  # keep track of optimal validation accuracy
    checkpt = time.time()

    results = np.zeros((nEpoch,10))

    oldRunMean, oldRunVar, oldParams = getVectorizedParams(net,device)

    print('%-9s %-9s %-11s %-11s %-9s %-11s %-9s %-9s %-9s %-9s' %
          ('epoch', 'time', '|runMean|', '|runVar|',
           'lr', '|params|', 'avgLoss', 'acc', 'valLoss', 'valAcc'),flush=True)
    for epoch in range(nEpoch):  # loop over the dataset multiple times



        # adjust learning rate by epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr.item(epoch)

        running_loss = 0.0
        # running_accuracy = 0.0
        running_num_correct = 0.0
        running_num_total   = 0.0

        ################ ADD RANDOM NOISE ########################
        # the unchanged tY and tTheta in net
        epsY      = np.random.uniform(-tYnoise, tYnoise, len(net.tY))
        epsTheta  = np.random.uniform(-tThetaNoise, tThetaNoise, len(net.tTheta))
        tYnew     = (net.tY + epsY).tolist()
        tThetaNew = (net.tTheta + epsTheta).tolist()
        # print('   new tY, tTheta: ',(' '.join(['{:.2f}']*len(tYnew))).format(*tYnew), ' , ',
        #       (' '.join(['{:.2f}'] * len(tThetaNew))).format(*tThetaNew))
        for idx in range(len(net.dynamicBlocks)):
            net.dynamicBlocks[idx].setTimeSteps(tY=tYnew, tTheta=tThetaNew)
        ############################################################


        for i, data in enumerate(loaderTrain, 0):

            net.train() # set model to train mode

            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)
            # forward + backward + optimize


            #x    = net(inputs,Kopen,Kresnet, Kconnect, Knorm, KtvNorm)
            x = net(inputs)
            loss, Si = misfit(x,labels)


            loss.backward()
            optimizer.step()

            _ , numCorrect, numTotal = getAccuracy(Si,labels)

            running_loss        += numTotal * loss.item()
            # running_accuracy    += accuracy
            running_num_correct += numCorrect
            running_num_total   += numTotal

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
                               #  Kopen, Kresnet, Kconnect, Knorm, KtvNorm, W)

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


        if valAcc > valOpt:
            # sPath = sBasePath + '_acc_' + "{:.3f}".format(valAcc)
            sPath = sBasePath + '_model'
            # saveParams(sPath+'.mat', Kopen, Kresnet, Kconnect, Knorm, KtvNorm, W, net.runMean, net.runVar)
            torch.save(net, sPath + '.pt')
            valOpt = valAcc # update
            sPathOpt = sPath


        checkpt = time.time()

        # end for batch in loader

        oldRunMean, oldRunVar, oldParams = newRunMean, newRunVar, newParams

    # end for epoch

    saveResults(sBasePath + '_results.mat', results)

    return sPathOpt

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

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


##### IS THIS USED ANYWHERE?
def trainNetReparametrization(net,optimizer,lr,nEpoch, device, sBasePath, loaderTrain,loaderVal, nMini=1, verbose=True,regularization_param=0.0001,checkpointing=False,reparametrization_length=None,tY_array=None,tTheta_array=None):

    sPathOpt = ''
    valOpt = 100000  # keep track of optimal validation loss
    checkpt = time.time()

    results = np.zeros((nEpoch,10))

    oldRunMean, oldRunVar, oldParams = getVectorizedParams(net,device)
    boundConstraints = clipWeights()

    time_since_reparametrization=0
    reparamtrization_time=0
    last_reparametrization=0
    print('%-9s %-9s %-11s %-11s %-9s %-11s %-9s %-9s %-9s %-9s' %
          ('epoch', 'time', '|runMean|', '|runVar|',
           'lr', '|params|', 'avgLoss', 'acc', 'valLoss', 'valAcc'), flush=True)

    for epoch in range(nEpoch):  # loop over the dataset multiple times

        # adjust learning rate by epoch

        #adjust parametrization
        if reparametrization_length is not None:
            time_since_reparametrization=time_since_reparametrization+1
            if reparametrization_length==time_since_reparametrization:
                time_since_reparametrization=0
                reparamtrization_time=reparamtrization_time+1
                if reparamtrization_time<len(tY_array)-1:
                    last_reparametrization = epoch
                    newtY=tY_array[min(reparamtrization_time,len(tY_array)-1)]
                    newtTheta = tTheta_array[min(reparamtrization_time, len(tTheta_array)-1)]

                    for idx in range(len(net.dynamicBlocks)):
                        net.dynamicBlocks[idx].setTimeSteps(tY=newtY, tTheta=newtTheta)

                    opt1 = copy.deepcopy(optimizer)
                    opt1.defaults = optimizer.defaults
                    opt1.param_groups = []

                    opt1.add_param_group({"params":net.parameters()})
                    optimizer=opt1
                    print("net tY:" + str(newtY))
                    print("net tTheta:" + str(newtTheta))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr.item(epoch-last_reparametrization)

        running_loss = 0.0
        # running_accuracy = 0.0
        running_num_correct = 0.0
        running_num_total   = 0.0
        for i, data in enumerate(loaderTrain, 0):

            net.train() # set model to train mode

            if checkpointing is True:
                loss, numCorrect, numTotal = net.checkpoint_train(data, optimizer, device,regularization_param)
            else:
            # get the inputs
                inputs, labels = data

            # zero the parameter gradients
                optimizer.zero_grad()

                inputs, labels = inputs.to(device), labels.to(device)
            # forward + backward + optimize

                x = net(inputs)
                loss, Si = misfit(x,net.W,labels,device)

            # add regularization to loss
                if regularization_param != 0:
                    loss =loss + regularization_param*net.regularization()
                loss.backward()
                optimizer.step()

                _ , numCorrect, numTotal = getAccuracy(Si,labels)

            running_loss        += numTotal * loss.item()
            # running_accuracy    += accuracy
            running_num_correct += numCorrect
            running_num_total   += numTotal

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
        if oldRunMean.size()== newRunMean.size():
            results[epoch, 2] = torch.norm(oldRunMean-newRunMean)
            results[epoch, 3] = torch.norm(oldRunVar - newRunVar)
        else:
            results[epoch, 2] =0
            results[epoch, 3] = 0
        results[epoch, 4] = optimizer.param_groups[0]['lr']
        if oldParams.size()== newParams.size():
            results[epoch, 5] = torch.norm(oldParams - newParams)
        else:
            results[epoch, 5] = 0
        results[epoch, 6] = running_loss / running_num_total
        results[epoch, 7] = running_num_correct * 100 / running_num_total
        results[epoch, 8] = valLoss
        results[epoch, 9] = valAcc * 100

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

        # end for batch in loader

        # impose bound constraints / clip the weights
        net.apply(boundConstraints)

        oldRunMean, oldRunVar, oldParams = newRunMean, newRunVar, newParams

    # end for epoch

    saveResults(sBasePath + '_results.mat', results)

    return sPathOpt

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------



# REMOVE THESE??? OUTDATED
# impose boundary constraints on the weights
# they must be between -1 and 1
class clipWeights(object):
    def __init__(self, min = -1, max=1):
        self.min = min
        self.max = max

    def __call__(self,module):
        # clip everything to [-1,1]
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(min=self.min, max=self.max)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                w = module.bias.data
                w.clamp_(min=self.min, max=self.max)
        # if hasattr(module, 'W'):
        #     w = module.W.data
        #     w.clamp_(min=self.min, max=self.max)

class clipConvs(object): # clip just the convolution weights and biases
    def __init__(self, min=-0.1, max=0.1):
        self.min = min
        self.max = max

    def __call__(self,module):
        if hasattr(module, 'conv'):
            module.conv.weight.data.clamp_(min=self.min, max=self.max)
            if module.conv.bias is not None:
                module.conv.bias.data.clamp_(min=self.min, max=self.max)

        if hasattr(module, 'conv1'):
            module.conv1.weight.data.clamp_(min=self.min, max=self.max)
            if module.conv1.bias is not None:
                module.conv1.bias.data.clamp_(min=self.min, max=self.max)

        if hasattr(module, 'conv2'):
            module.conv2.weight.data.clamp_(min=self.min, max=self.max)
            if module.conv2.bias is not None:
                module.conv2.bias.data.clamp_(min=self.min, max=self.max)



# For all run functions
def getRunDefaults(sDataset, gpu=0):
    # vFeat, transformTrain, transformTest, batchSize, lr, tTheta, device = getRunDefaults(sDataset)
    # sDataset - 'cifar10' , 'cifar100', 'stl10'
    # gpu - None means use cpu; otherwise, use an int to signify which gpu to use


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


