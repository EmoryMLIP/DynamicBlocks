# ClippedModule.py
# 5/2/19

from utils import *
import torch.nn as nn

class ClippedModule(nn.Module):
    """
    Extend nn.Module to include max and min values for bound constraints / clipping
    """

    def __init__(self): # include ability to set at initialization?
        super().__init__()
        self.minConv =  -0.5
        self.maxConv =   0.5
        self.minDef  =  -1.5
        self.maxDef  =   1.5
        # setClipValues() # use the defaults set in setClipValues


    def setClipValues(self, minConv = -0.5,maxConv=0.5, minDef=-1.5, maxDef=1.5):
        self.minConv = minConv
        self.maxConv = maxConv
        self.minDef  = minDef
        self.maxDef  = maxDef

    def calcClipValues(self,h,nPixels,nChan):

        # devlop some multiplier to adjust the defaults
        mult =1/h # larger time step, reduce the constraints
        mult = mult / (math.sqrt(nPixels))
        mult = mult * (500/ nChan**2)

        minConv = -1
        maxConv =  1
        self.setClipValues( minConv=mult*minConv, maxConv=mult*maxConv, minDef=-1.5, maxDef=1.5)




    def clip(self):
        # assume conv range is subset of default range

        if hasattr(self,'conv'):
            self.conv.weight.data.clamp_(min=self.minConv, max=self.maxConv)
            if self.conv.bias is not None:
                self.conv.bias.data.clamp_(min=self.minConv, max=self.maxConv)

        if hasattr(self, 'conv1'):
            self.conv1.weight.data.clamp_(min=self.minConv, max=self.maxConv)
            if self.conv1.bias is not None:
                self.conv1.bias.data.clamp_(min=self.minConv, max=self.maxConv)

        if hasattr(self, 'conv2'):
            self.conv2.weight.data.clamp_(min=self.minConv, max=self.maxConv)
            if self.conv2.bias is not None:
                self.conv2.bias.data.clamp_(min=self.minConv, max=self.maxConv)

        if hasattr(self, 'weight'):
            w = self.weight.data
            w.clamp_(min=self.minDef, max=self.maxDef)
        # if hasattr(module, 'bias'): # bias can be quite large( 3.0+ after removal of batch norm)
        #     if module.bias is not None:
        #         w = module.bias.data
        #         w.clamp_(min=self.min, max=self.max)

        for module in self.children(): # just immediate children
            if hasattr(module, 'clip'):
                module.clip()
            else: # this may not be robust...only goes two levels down...trying to get to the normLayers
                if hasattr(module, 'weight'):
                    w = module.weight.data
                    w.clamp_(min=self.minDef, max=self.maxDef)
                # if hasattr(module, 'bias'): # bias can be quite large( 3.0+ after removal of batch norm)
                #     if module.bias is not None:
                #         w = module.bias.data
                #         w.clamp_(min=self.min, max=self.max)
                for child in module.children():
                    if hasattr(child, 'clip'):
                        child.clip()


        return self



if __name__ == "__main__":
    from modules.DoubleLayer import *
    from RKNet import *

    # mini-test

    l = DoubleLayer([4,4], {'normLayer':nn.BatchNorm2d(4) } )

    l.setClipValues( minConv=-0.1, maxConv=0.1, minDef=-1.5, maxDef=1.5)
    l.conv1.weight.data =  l.conv1.weight.data * 0 + 4
    l.conv1.bias.data   = l.conv1.bias.data * 0 + 4
    l.conv2.weight.data =  l.conv2.weight.data * 0 + 4
    l.conv2.bias.data   = l.conv2.bias.data * 0 + 7
    l.normLayer1.weight.data = l.normLayer1.weight.data * 0 + 5
    l.normLayer2.weight.data = l.normLayer2.weight.data * 0 + 2
    l.normLayer1.bias.data   = l.normLayer1.bias.data * 0 + 3
    l.normLayer2.bias.data   = l.normLayer2.bias.data * 0

    l.clip()

    for n, p in l.named_parameters():
        print(n, p.data)
        # assert(torch.max(p.data) == ?????) maxConv or maxDef


    l.conv1.weight.data =  l.conv1.weight.data * 0 - 4
    l.conv1.bias.data   = l.conv1.bias.data * 0 - 4
    l.conv2.weight.data =  l.conv2.weight.data * 0 - 4
    l.conv2.bias.data   = l.conv2.bias.data * 0 - 4
    l.normLayer1.weight.data = l.normLayer1.weight.data * 0 - 4
    l.normLayer2.weight.data = l.normLayer2.weight.data * 0 - 4
    l.normLayer1.bias.data = l.normLayer1.bias.data * 0 - 4
    l.normLayer1.bias.data = l.normLayer1.bias.data * 0 - 4

    l.clip()

    for n, p in l.named_parameters():
        print(n, p.data)
        # assert(torch.max(p.data) == ?????) maxConv or maxDef


    # -----------------------------------------
    net = RKNet(tY=[0,1,2,3,4], tTheta=[0,1,2,3,4], nChanIn=3,
                nClasses=10, vFeat=[4,7,7], dynamicScheme = rk1, layer = DoubleSymLayer)

    for n, p in net.named_parameters():
        p.data = p.data*0 + 5


    net.clip()

    for n, p in net.named_parameters():
        if torch.max(p.data) > 4:
            print(n, p.data)

    print('code finished running')

