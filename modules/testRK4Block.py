from modules.rk4 import *
from utils import *
from modules.DoubleSymLayer import DoubleSymLayer




torch.set_printoptions(precision=10)


def DoubleSymLayerBatchNorm(x, K):
    z = conv3x3(x, K)
    z = F.batch_norm(z, running_mean=torch.zeros(K.size(0)),
                     running_var=torch.ones(K.size(0)), training=True)
    z = F.relu(z)
    z = - convt3x3(z, K)
    return z


# OLD one
def rk4DoubleSymBlock(x,Kresnet, tTheta, tY, act=F.relu, weightBias = [None] , tvnorm=True):
    """
    Implementation of a Runge-Kutta 4 block that uses the DoubleSym Layer at each of the steps
    """
    nt = len(tY)-1

    for k in range(nt):
        tYk = tY[k]
        hi = tY[k + 1] - tYk

        if weightBias[0] is not None:   # really should check if all are None
            interpWeight = inter1D(weightBias[0::2], tTheta, [tYk, tYk + hi / 2, tYk + hi])  # scaling weight
            interpBias = inter1D(weightBias[1::2], tTheta, [tYk, tYk + hi / 2, tYk + hi])  # bias
        else:
            interpWeight = [None,None,None]
            interpBias   = [None,None,None]

        interpK  = inter1D(Kresnet, tTheta, [tYk , tYk+hi/2 , tYk+hi ])

        # first intermediate step
        z1 = DoubleSymLayerBatchNorm( x , interpK[0] )

        # second intermediate step
        z2 = DoubleSymLayerBatchNorm( x + z1*(hi/2) , interpK[1])

        # third intermediate step
        z3 = DoubleSymLayerBatchNorm( x + z2*(hi/2) , interpK[1]  )

        # fourth intermediate
        z4 = DoubleSymLayerBatchNorm( x + z3*hi , interpK[2] )

        x = x + hi*(z1 + 2*z2 + 2*z3 + z4)/6

    return x


# if __name__ == "__main__":

tTheta = [0 , 2]
tY = [0,1,2]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
nChan = 2
nClasses = 5
x = normalInit([10 , nChan , 32 , 32]).to(device)  # (numImages, numChans, image height, image width)


labels = torch.LongTensor([1 ,2 ,3 ,4 ,3 ,2 ,1 ,0 ,2 ,3]).to(device)
Kconnect = normalInit([nChan, nChan, 1, 1]).to(device)



# dsLayer = DoubleSymLayer( [nChan,7] , params={'norm': nn.BatchNorm2d(nChan)})
dsLayer = DoubleSymLayer
layerParams = { 'vFeat': [nChan ,nChan+1] ,
                'act' : nn.ReLU(),
                'normLayer': nn.BatchNorm2d(nChan+1)}
# layerParams = {'vFeat': [ nChan,3], 'act': nn.ReLU()}
net = rk4( tTheta, tY, dsLayer, layerParams=layerParams)
# for old method
origWeights =[]
for i in range(len(net.controlLayers)):
    origWeights.append(net.controlLayers[i].weight.data.clone())


net.to(device)

stdv = 0.1
W = stdv * torch.randn(nChan +1, nClasses).to(device)  # plus one for the bias
optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9, weight_decay=0, nesterov=False)

optimizer.zero_grad()
y1 = net.forward(x)
y3 = F.avg_pool2d(y1, x.shape[2:4])

loss1, _ = misfitW(y3, W, labels, device)
loss1.backward()
optimizer.step()


# --------------

# use same initializations
K = []
for i in range(len(origWeights)):
    K.append( nn.Parameter(origWeights[i].clone()) )

params = []
nWeights = 0
params, nWeights = appendTensorList(params, K, nWeights)
optimizer2 = optim.SGD(params, lr=1, momentum=0.9, weight_decay=0, nesterov=False)

# for i in range(len(K)):
#     K[i] = K[i].to(device)

optimizer2.zero_grad()

class compareNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x, K, tTheta, tY):
        y2 = rk4DoubleSymBlock(x, K, tTheta, tY, act=F.relu, weightBias=[None], tvnorm=False)

        y4 = F.avg_pool2d(y2, x.shape[2:4])

        return y2,y4

net2 = compareNet()
y2,y4 = net2.forward(x,K,tTheta,tY)

loss2, _ = misfitW(y4, W, labels, device)
loss2.backward()
optimizer2.step()


# print('block 2-norm difference:', torch.norm(y2 - y1, p=2).data)       # want = 0
# print('net   2-norm difference:', torch.norm(y4 - y3, p=2).data)       # want = 0
# print('loss  2-norm difference:', torch.norm(loss2 - loss1, p=2).data) # want = 0
# print('K     2-norm difference:', listNorm([net.controlLayers[i].weight for i in range(len(net.controlLayers))] ,K))
#                                                                        # want = 0
# print('K update:               ', listNorm(origWeights ,K))            # want > 0

tol = 1e-4
assert(torch.norm(y2 - y1, p=2).data < tol)
assert(torch.norm(y4 - y3, p=2).data < tol)
assert(torch.norm(loss2 - loss1, p=2).data < tol)
assert( listNorm([net.controlLayers[i].weight for i in range(len(net.controlLayers))] ,K) < tol)

assert( listNorm(origWeights ,K) > 1e-4)            # want > 0
print('tests passed')









# # make auto-differentiation graph
# from graphModel import make_dot
#
# g = make_dot(y3, net.state_dict())
# g.view()


# # see all the parameters.....net.parameters() is a generator
# print('\nparameters of the model:\n')
# for i in enumerate(net.parameters()):
#     print(i)
