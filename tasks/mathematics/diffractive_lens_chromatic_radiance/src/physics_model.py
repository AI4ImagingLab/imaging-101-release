"""Local patchwise diffraction blur and transfer operators.

Adapted from Sun/Peng/Heidrich ICCV2017 code under CC BY-NC4.0.
"""
import numpy as np

def positions(shape,size,step):
    """MATLAB MEX ordering: column outer, row inner, last patch flush to edge."""
    axes=[]
    for n in shape:
        if n<size:raise ValueError('patch exceeds image')
        a=list(range(0,n-size+1,step))
        if a[-1]!=n-size:a.append(n-size)
        axes.append(a)
    return [(i,j) for j in axes[1] for i in axes[0]]

def patch_columns(image,size,step):
    return np.column_stack([image[i:i+size,j:j+size].ravel(order='F') for i,j in positions(image.shape,size,step)])

def overlap(columns,shape,size,step):
    out=np.zeros(shape,dtype=np.float64)
    for k,(i,j) in enumerate(positions(shape,size,step)):out[i:i+size,j:j+size]+=columns[:,k].reshape((size,size),order='F')
    return out

def gradient(image):
    """MATLAB intermediate differences with replicated exterior boundary."""
    gx=np.zeros_like(image);gy=np.zeros_like(image)
    gx[:,:-1]=np.diff(image,axis=1);gy[:-1]=np.diff(image,axis=0)
    return gx,gy

def window(size):return np.outer(np.hamming(size),np.hamming(size))

def weights(shape,size,step):
    w=window(size);total=np.zeros(shape)
    for i,j in positions(shape,size,step):total[i:i+size,j:j+size]+=w
    return w,total

def psf_prior(size,alpha,beta):
    dx=np.zeros((size,size));dy=dx.copy();dx[0,:2]=[1,-1];dy[:2,0]=[1,-1]
    return beta+alpha*(abs(np.fft.fft2(dx))**2+abs(np.fft.fft2(dy))**2)

def estimate_psf(sharp,blur,prior):
    x=np.fft.fft2(sharp);y=np.fft.fft2(blur)
    return np.conj(x)*y/(np.conj(x)*x+prior)

def features(image,gx,gy,i,j,size):
    return np.column_stack((np.ones(size*size),image[i:i+size,j:j+size].ravel(order='F'),gx[i:i+size,j:j+size].ravel(order='F'),gy[i:i+size,j:j+size].ravel(order='F')))
