
from modules.DoubleSymLayer import DoubleSymLayer

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
paramsStruct = {'normLayer': nn.BatchNorm2d(num_features=4),
                'conv': nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1, stride=1)}

net = DoubleSymLayer(vFeat, params=paramsStruct)
net.to(device)
origK = net.weight.data.clone()  # for old method
K = nn.Parameter(origK.clone())
optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=0, nesterov=False)

optimizer.zero_grad()
y1 = net.forward(x)
y1 = F.avg_pool2d(y1, x.shape[2:4])

loss1, _ = misfitW(y1, W, labels, device)
loss1.backward()
optimizer.step()


# check that both convolutions point to the same parameter
# print(net.convt.weight is net.conv.weight)         # true when pointing to the same object
# print(net.convt.weight is net.conv.weight.clone()) # false because different objects
# print(net.convt.weight==net.conv.weight.clone()) # all true because each element is the same value

assert(net.convt.weight is net.conv.weight)
assert(not  net.convt.weight is net.conv.weight.clone() )


# ----------------------------------------------------------------------
def compareFunc(x, K ,device): # functional DoubleSymLayer
    z = conv3x3(x, K)
    z = F.batch_norm(z, running_mean=torch.zeros(K.size(0) ,device=device),
                     running_var=torch.ones(K.size(0) ,device=device), training=True)
    z = F.relu(z)
    z = - convt3x3(z, K)
    return z
# old method
optimParams = [{'params': K}]
nWeights = 0
optimizer = optim.SGD(optimParams, lr=1e-1, momentum=0.9, weight_decay=0, nesterov=False)

optimizer.zero_grad()

y2 = compareFunc(x, K, device)

y2 = F.avg_pool2d(y2, x.shape[2:4])

loss2, _ = misfitW(y2, W, labels, device)
loss2.backward()
optimizer.step()

# ----------------------------------------------------------------------

# print('layer 2-norm difference:', torch.norm(y2 - y1, p=2).data)                   # want = 0
# print('loss 2-norm difference: ', torch.norm(loss2 - loss1, p=2).data)             # want = 0
# print('K    2-norm difference: ', torch.norm(net.weight.data - K.data, p=2).data)  # want = 0
# print('K update:               ', torch.norm(origK - K.data, p=2).data)            # want > 0

tol = 1e-5
assert(torch.norm(y2 - y1, p=2).data < tol)
assert(torch.norm(loss2 - loss1, p=2).data < tol)
assert(torch.norm(net.weight.data - K.data, p=2).data < tol)

assert(torch.norm(origK - K.data, p=2).data > 1e-4)            # want > 0
print('tests passed')


