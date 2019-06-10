
from modules.PreactDoubleLayer import PreactDoubleLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vFeat = [4, 4]
nChan = vFeat[0]
nClasses = 5

# random batch
x = normalInit([10, nChan, 32, 32]).to(device)  # (numImages, numChans, image height, image width)
W = normalInit([nChan + 1, nClasses]).to(device)  # plus one for the bias
labels = torch.LongTensor([1, 2, 3, 4, 3, 2, 1, 0, 2, 3]).to(device)
Kconnect = normalInit([nChan, nChan, 1, 1]).to(device)

# ----------------------------------------------------------------------
# new approach
paramsStruct = {'normLayer1': nn.BatchNorm2d(num_features=nChan),
                'normLayer2': nn.BatchNorm2d(num_features=nChan),
                'conv1': nn.Conv2d(in_channels=nChan, out_channels=nChan, kernel_size=3, padding=1, stride=1),
                'conv2': nn.Conv2d(in_channels=nChan, out_channels=nChan, kernel_size=3, padding=1, stride=1)}

net = PreactDoubleLayer(vFeat, params=paramsStruct)
net.to(device)
origK1 = net.conv1.weight.data.clone()  # for old method
origK2 = net.conv2.weight.data.clone()  # for old method
K1 = nn.Parameter(origK1.clone())
K2 = nn.Parameter(origK2.clone())
optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=0, nesterov=False)

optimizer.zero_grad()
y1 = net.forward(x)
y1 = F.avg_pool2d(y1, x.shape[2:4])

loss1, _ = misfitW(y1, W, labels, device)
loss1.backward()
optimizer.step()

# ----------------------------------------------------------------------
def compareFunc(x, K1, K2 ,device): # functional Preactivated DoubleLayer
    z = F.relu(x)
    z = conv3x3(z, K1)
    z = F.batch_norm(z, running_mean=torch.zeros(K1.size(0) ,device=device),
                     running_var=torch.ones(K1.size(0) ,device=device), training=True)
    z = F.relu(z)
    z = conv3x3(z, K2)
    z = F.batch_norm(z, running_mean=torch.zeros(K2.size(0), device=device),
                     running_var=torch.ones(K2.size(0), device=device), training=True)
    return z
# old method
optimParams = [{'params': K1}, {'params':K2}]
nWeights = 0
optimizer = optim.SGD(optimParams, lr=1e-1, momentum=0.9, weight_decay=0, nesterov=False)

optimizer.zero_grad()

y2 = compareFunc(x, K1,K2, device)

y2 = F.avg_pool2d(y2, x.shape[2:4])

loss2, _ = misfitW(y2, W, labels, device)
loss2.backward()
optimizer.step()

# ----------------------------------------------------------------------

# print('layer 2-norm difference:', torch.norm(y2 - y1, p=2).data)                         # want = 0
# print('loss 2-norm difference: ', torch.norm(loss2 - loss1, p=2).data)                   # want = 0
# print('K1    2-norm difference:', torch.norm(net.conv1.weight.data - K1.data, p=2).data) # want = 0
# print('K2    2-norm difference:', torch.norm(net.conv2.weight.data - K2.data, p=2).data) # want = 0
# print('K1 update:              ',torch.norm(origK1 - K1.data, p=2).data)                 # want > 0
# print('K2 update:              ',torch.norm(origK2 - K2.data, p=2).data)                 # want > 0

tol = 1e-5
assert(torch.norm(y2 - y1, p=2).data < tol)
assert(torch.norm(loss2 - loss1, p=2).data < tol)
assert( torch.norm(net.conv1.weight.data - K1.data, p=2).data < tol )
assert( torch.norm(net.conv2.weight.data - K2.data, p=2).data < tol )

assert( torch.norm(origK1 - K1.data, p=2).data > 1e-4)
assert( torch.norm(origK2 - K2.data, p=2).data > 1e-4)
print('tests passed')
