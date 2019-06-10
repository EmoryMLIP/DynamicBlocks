
from modules.ConnectingLayer import ConnectingLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vFeat = [3, 4]
nChan = vFeat[0]
nClasses = 5

# random batch
x = normalInit([10, 3, 32, 32]).to(device)  # (numImages, numChans, image height, image width)
W = normalInit([nChan + 1, nClasses]).to(device)  # plus one for the bias
labels = torch.LongTensor([1, 2, 3, 4, 3, 2, 1, 0, 2, 3]).to(device)
Kconnect = normalInit([nChan, nChan, 1, 1]).to(device)

# ----------------------------------------------------------------------
# new approach
paramsStruct = {'normLayer': nn.BatchNorm2d(num_features=vFeat[1]),
                'conv': nn.Conv2d(in_channels=vFeat[0], out_channels=vFeat[1], kernel_size=3, padding=1, stride=1)}

net = ConnectingLayer(vFeat, params=paramsStruct)
net.to(device)
origK = net.conv.weight.data.clone().to(device)  # for old method
K = nn.Parameter(origK.clone()).to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=0, nesterov=False)

optimizer.zero_grad()
y1 = net.forward(x)
y1 = F.avg_pool2d(y1, x.shape[2:4])

loss1, _ = misfitW(y1, W, labels, device)
loss1.backward()
optimizer.step()

# ----------------------------------------------------------------------
def compareFunc(x, Kopen, normFlag='batch', eps=1e-5, device=torch.device('cpu')):
    x = conv3x3(x, Kopen)
    if normFlag is 'batch':
        x = F.batch_norm(x,
                         running_mean = torch.zeros(Kopen.size(0)).to(device),
                         running_var  = torch.ones(Kopen.size(0)).to(device),
                         weight       = torch.ones(Kopen.size(0)).to(device),
                         bias         = torch.zeros(Kopen.size(0)).to(device),
                         training = True, eps=eps)
    elif normFlag is 'instance':
        x = F.instance_norm(x, weight=weightBias[0], bias=weightBias[1])
    x = F.relu(x)
    return x
# old method
optimParams = [{'params': K}]
nWeights = 0
optimizer = optim.SGD(optimParams, lr=1e-1, momentum=0.9, weight_decay=0, nesterov=False)

optimizer.zero_grad()

y2 = compareFunc(x, K, device=device)

y2 = F.avg_pool2d(y2, x.shape[2:4])

loss2, _ = misfitW(y2, W, labels, device)
loss2.backward()
optimizer.step()

# ----------------------------------------------------------------------

# print('layer 2-norm difference:', torch.norm(y2 - y1, p=2).data)
# print('loss 2-norm difference: ', torch.norm(loss2 - loss1, p=2).data)
# print('K    2-norm difference: ', torch.norm(net.conv.weight.data - K.data, p=2).data)
# print('K update:               ', torch.norm(origK - K.data, p=2).data)


tol = 1e-7
assert(torch.norm(y2 - y1, p=2).data < tol)
assert(torch.norm(loss2 - loss1, p=2).data < tol)
assert(torch.norm(net.conv.weight.data - K.data, p=2).data < tol)

assert( torch.norm(origK - K.data, p=2).data > 1e-4)            # want > 0
print('tests passed')



