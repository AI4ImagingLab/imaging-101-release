"""Explicit portable Joseph arithmetic diagnostic, not the standard GPU backend.

This pure CPU matrix follows the same ray equations and 1/256 texture weight
quantization, but its NumPy float32 arithmetic can cross a different texture bin
than native CUDA. The recorded full-size difference is about3e-5 relative.
Do not substitute this diagnostic into the standard inverse or claim exact GPU
parity. It supports independent CPU forward/adjoint tests and error analysis.
"""
import numpy as np
from scipy.sparse import coo_matrix

def joseph_matrix(shape,angles,rays,texture_quantization=True):
    m,n=shape;rr=[];cc=[];vv=[]
    for ai,angle in enumerate(angles):
        theta=np.float32(np.deg2rad(angle));c=np.float32(np.cos(theta));s=np.float32(np.sin(theta))
        fma=lambda a,b,c:np.asarray(np.asarray(a,dtype='float64')*np.asarray(b,dtype='float64')+np.asarray(c,dtype='float64'),dtype='float32')
        dsx=np.float32(-rays/2*float(c));dsy=np.float32(-rays/2*float(s));sx=fma(.5,c,dsx);sy=fma(.5,s,dsy);ray=np.arange(rays,dtype='float32');detx=fma(ray,c,sx);dety=fma(ray,s,sy)
        primary=m if abs(c)>=abs(s) else n;secondary=n if abs(c)>=abs(s) else m
        slope=np.float32(-s/c) if abs(c)>=abs(s) else np.float32(-c/s)
        intercept=fma(-slope,dety,detx) if abs(c)>=abs(s) else fma(-slope,detx,dety)
        position=np.empty((rays,primary),dtype='float32')
        for start in range(0,primary,32):
            coordinate=fma(slope,np.float32(start-primary/2+.5),intercept)
            coordinate=np.float32(np.float32(np.float32(coordinate+np.float32(secondary/2))-.5)+.5)
            for index_axis in range(start,min(start+32,primary)):
                position[:,index_axis]=np.float32(coordinate-.5);coordinate=np.float32(coordinate+slope)
        index=np.floor(position).astype('int32');fraction=position-index.astype('float32');step=np.float32(np.sqrt(np.float32(slope*slope+1)))
        if abs(c)>=abs(s):
            base=np.broadcast_to(np.arange(m,dtype='int32')[None,:]*n,position.shape);cols=[base+index,base+index+1];valid=[(index>=0)&(index<n),(index+1>=0)&(index+1<n)]
        else:
            base=np.broadcast_to(np.arange(n,dtype='int32')[None,:],position.shape);cols=[index*n+base,(index+1)*n+base];valid=[(index>=0)&(index<m),(index+1>=0)&(index+1<m)]
        if texture_quantization:fraction=np.rint(fraction*256).astype('float32')/np.float32(256)
        rows=np.broadcast_to((ai*rays+np.arange(rays,dtype='int32'))[:,None],position.shape)
        for col,good,weight in zip(cols,valid,[(1-fraction)*step,fraction*step]):
            good=good&(weight>0);rr.append(rows[good]);cc.append(col[good]);vv.append(weight[good])
    return coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(len(angles)*rays,m*n)).tocsr()
