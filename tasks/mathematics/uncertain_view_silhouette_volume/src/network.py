"""Complete selected five-scale 3-D deep prior, built from general layers.

Adapted from the MIT-licensed ShapeRecDeepPrior skip3d architecture. Module
construction order preserves the original CPU parameter initialization RNG.
No pretrained model or upstream framework is used.
"""
import torch
from torch import nn

class Concatenate(nn.Module):
    """Evaluate two branches on the same input and concatenate channels."""
    def __init__(self,skip,deeper):
        super().__init__();self.skip=skip;self.deeper=deeper
    def forward(self,x):return torch.cat((self.skip(x),self.deeper(x)),dim=1)

def add(sequence,module):sequence.add_module(str(len(sequence)+1),module)
def conv(cin,cout,kernel,stride=1):
    return nn.Sequential(nn.Conv3d(cin,cout,kernel,stride,padding=(kernel-1)//2,bias=True))
def activation():return nn.LeakyReLU(.2,inplace=True)

def build_network():
    """Return five-scale 3input/1output encoder-decoder with sigmoid density."""
    down=[8,16,32,64,128];up=[8,16,32,64,128];skip_channels=[0,0,0,4,4]
    model=nn.Sequential();current=model;input_channels=3
    for i in range(5):
        deeper=nn.Sequential();skip=nn.Sequential();sk=skip_channels[i]
        add(current,Concatenate(skip,deeper) if sk else deeper)
        add(current,nn.BatchNorm3d(sk+(up[i+1] if i<4 else down[i])))
        if sk:
            add(skip,conv(input_channels,sk,1));add(skip,nn.BatchNorm3d(sk));add(skip,activation())
        add(deeper,conv(input_channels,down[i],3,2));add(deeper,nn.BatchNorm3d(down[i]));add(deeper,activation())
        add(deeper,conv(down[i],down[i],3));add(deeper,nn.BatchNorm3d(down[i]));add(deeper,activation())
        nested=nn.Sequential()
        if i<4:add(deeper,nested)
        k=up[i+1] if i<4 else down[i]
        add(deeper,nn.Upsample(scale_factor=2,mode='trilinear'))
        add(current,conv(sk+k,up[i],3));add(current,nn.BatchNorm3d(up[i]));add(current,activation())
        add(current,conv(up[i],up[i],1));add(current,nn.BatchNorm3d(up[i]));add(current,activation())
        input_channels=down[i];current=nested
    add(model,conv(up[0],1,1));add(model,nn.Sigmoid())
    return model
