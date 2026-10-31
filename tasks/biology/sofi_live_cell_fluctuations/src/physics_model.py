"""Local SOFI cumulants; adapted from PySOFI, MIT (see LICENSE)."""
import math
import numpy as np

def fluorescence_image(emitter_images,blinking):
    """(emitter,y,x) PSF images and (frame,emitter) on/off -> frame ADU."""
    return np.einsum('te,eyx->tyx',blinking,emitter_images)

def moments(stack,mean,highest_order=4):
    """Central moments, including original int64 truncation once per order."""
    result={}
    for order in range(1,highest_order+1):
        value=np.zeros(mean.shape,dtype=np.float64)
        for frame in stack:value+=np.power(frame-mean,order)
        result[order]=(value/len(stack)).astype(np.int64)
    return result

def cumulants(moment_set):
    """Recursive central cumulants, preserving signs and float64 arithmetic."""
    result={}
    for order in range(1,len(moment_set)+1):
        terms=[math.comb(order-1,i)*result[order-i]*moment_set[i] for i in range(1,order)]
        result[order]=moment_set[order]-np.sum(np.array(terms),axis=0)
    return result
