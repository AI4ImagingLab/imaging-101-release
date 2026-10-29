"""hdrplus burst alignment functions.
Copyright (c) 2021 Antoine Monod

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.

This file implements an algorithm possibly linked to the patent US9077913B2.
This file is made available for the exclusive aim of serving as scientific tool
to verify the soundness and completeness of the algorithm description.
Compilation, execution and redistribution of this file may violate patents rights in certain countries.
The situation being different for every country and changing over time,
it is your responsibility to determine which patent rights restrictions apply to you
before you compile, use, modify, or redistribute this file.
A patent lawyer is qualified to make this determination.
If and only if they don't conflict with any patent terms,
you can benefit from the following license terms attached to this file.

Adapted locally: explicit NumPy distances, corrected 3x3 indexing and safe RMS.
"""
import numpy as np
from .preprocessing import getTiles
def computeTilesDistanceL1_(refTiles,altTiles,offsets):
    h,w=offsets.shape[:2];size=refTiles.shape[-1]
    ri=np.arange(h)[:,None]*(size//2);rj=np.arange(w)[None,:]*(size//2)
    rounded=np.trunc(offsets+np.where(offsets>=0,.5,-.5)).astype(int)
    ai=np.clip(ri+rounded[:,:,0],0,altTiles.shape[0]-1);aj=np.clip(rj+rounded[:,:,1],0,altTiles.shape[1]-1)
    diff=refTiles[ri,rj]-altTiles[ai,aj]
    return np.sum(abs(diff).astype('float64'),axis=(-2,-1)).astype('float32')
def computeDistance(refPatch,searchArea,distance='L2'):
    h,w,size,_=refPatch.shape;side=searchArea.shape[-1]-size+1
    result=np.empty((h,w,side,side),dtype='float32')
    for i in range(side):
        for j in range(side):
            diff=searchArea[:,:,i:i+size,j:j+size]-refPatch
            # Numba original subtracts float32, accumulates double precision.
            values=abs(diff).astype('float64') if distance=='L1' else (diff*diff).astype('float64')
            result[:,:,i,j]=values.sum(axis=(-2,-1))
    return result.reshape(h*w,side*side)
def subPixelMinimum(arrDst,arrIdx):
    side=int(np.sqrt(arrDst.shape[1]));idx=arrIdx.astype(int)
    delta=np.array([(i-1)*side+j-1 for i in range(3) for j in range(3)])
    d=arrDst[np.arange(len(idx))[:,None],idx[:,None]+delta].astype('float64')
    filters=np.array([[.25,-.5,.25,.5,-1,.5,.25,-.5,.25],[.25,.5,.25,-.5,-1,-.5,.25,.5,.25],[.25,0,-.25,0,0,0,-.25,0,.25],[-.125,0,.125,-.25,0,.25,-.125,0,.125],[-.125,-.25,-.125,0,0,0,.125,.25,.125]])
    # Sequential inner-product order matches the original compiled loop.
    v=np.zeros((len(idx),5),dtype='float64')
    for k in range(9):v+=d[:,k,None]*filters[None,:,k]
    a11=np.maximum(v[:,0],0);a22=np.maximum(v[:,1],0);a12=v[:,2].copy();a12[a11*a22-a12*a12<0]=0
    det=a11*a22-a12*a12;out=np.zeros((len(idx),2),dtype='float64');good=det!=0
    out[good,0]=-(a11[good]*v[good,4]-a12[good]*v[good,3])/det[good]
    out[good,1]=-(a22[good]*v[good,3]-a12[good]*v[good,4])/det[good]
    out[np.linalg.norm(out,axis=1)>=1]=0
    return out.astype(arrDst.dtype)
def sample_forward(image,row_offsets,column_offsets):
    """Integer pullback measurement model, clipped at finite detector boundaries."""
    yy,xx=np.indices(image.shape)
    iy=np.clip(yy+row_offsets,0,image.shape[0]-1).astype(int);ix=np.clip(xx+column_offsets,0,image.shape[1]-1).astype(int)
    return image[iy,ix]
def sample_adjoint(values,row_offsets,column_offsets):
    yy,xx=np.indices(values.shape);iy=np.clip(yy+row_offsets,0,values.shape[0]-1).astype(int);ix=np.clip(xx+column_offsets,0,values.shape[1]-1).astype(int)
    result=np.zeros_like(values);np.add.at(result,(iy,ix),values);return result
