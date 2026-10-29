"""Explicit factorized Compton response, transpose and equal-area ring geometry.

No spherical-map or imaging reconstruction package is imported.
"""
import numpy as np
from scipy.sparse import coo_matrix

NSIDE=16
NPIX=12*NSIDE**2
PIXEL_AREA=4*np.pi/NPIX
NUMERICAL_ZERO=1e-12

class ComptonOperator:
    """Factorized attitude rotation then Compton detector response.

    Coordinates index (attitude, Galactic pixel, local incoming direction).
    Their weights are seconds. Response entries are square centimetres.
    The explicit solid-angle factor converts intensity to expected counts.
    """
    def __init__(self,response,coordinates,weights,exposure=None,pixel_area=PIXEL_AREA,response_area=1.):
        self.response=np.asarray(response)
        self.response_area=np.float32(response_area)
        self.nlocal,self.ndata=self.response.shape
        coordinates=np.asarray(coordinates)
        self.natt=int(coordinates[0].max())+1
        self.nsky=int(coordinates[1].max())+1
        matrix=coo_matrix((np.asarray(weights,dtype=np.float32),
                          (coordinates[0]*self.nlocal+coordinates[2],coordinates[1])),
                         shape=(self.natt*self.nlocal,self.nsky))
        self.rotation=matrix.tocsr()
        self.pixel_area=pixel_area
        self.exposure=None if exposure is None else np.asarray(exposure,dtype=np.float32).reshape(-1)

    def forward(self,intensity,add_floor=True):
        """Return (attitude, measured-bin) expected source photon counts."""
        rotated=(self.rotation@np.asarray(intensity,dtype=np.float32).reshape(-1)).reshape(self.natt,self.nlocal)
        expectation=np.empty((self.natt,self.ndata),dtype=np.float32)
        # Partition output columns only: every dot product still uses all
        # local directions. Integer response storage avoids a second 2GB copy.
        for start in range(0,self.ndata,8192):
            stop=min(start+8192,self.ndata)
            block=np.asarray(self.response[:,start:stop],dtype=np.float32)*self.response_area
            expectation[:,start:stop]=rotated@block
        expectation*=self.pixel_area
        if add_floor:expectation+=NUMERICAL_ZERO
        return expectation

    def adjoint(self,data):
        """Transpose of the count operator, excluding its numerical floor."""
        values=np.asarray(data,dtype=np.float32).reshape(self.natt,self.ndata)
        local=np.empty((self.natt,self.nlocal),dtype=np.float32)
        # Partition output incoming-direction columns, never the reduction
        # dimension: each transpose product uses the whole detector histogram.
        for start in range(0,self.nlocal,32):
            stop=min(start+32,self.nlocal)
            block=np.asarray(self.response[start:stop],dtype=np.float32)*self.response_area
            local[:,start:stop]=values@block.T
        result=self.rotation.T@local.reshape(-1)
        result*=self.pixel_area
        return np.asarray(result,dtype=np.float32)

    def compute_exposure(self):
        """Exposure Aᵀ1 in cm² s sr, with original response-sum convention."""
        local_sum=np.empty(self.nlocal,dtype=np.float32)
        for start in range(0,self.nlocal,32):
            stop=min(start+32,self.nlocal)
            block=np.asarray(self.response[start:stop],dtype=np.float32)*self.response_area
            local_sum[start:stop]=block.sum(axis=1)
        # Summing attitude weights before contraction matches the source's
        # factorized exposure definition; no target field is used.
        coo=self.rotation.tocoo()
        combined=coo_matrix((coo.data,(coo.col,coo.row%self.nlocal)),shape=(self.nsky,self.nlocal)).tocsr()
        result=combined@local_sum
        result*=self.pixel_area
        return np.asarray(result,dtype=np.float32)

def ring_pixel_angles(nside=NSIDE):
    """Return colatitude/longitude radians in equal-area HEALPix RING order."""
    theta=[];phi=[]
    for ring in range(1,4*nside):
        if ring<nside:
            nr=ring;z=1-ring**2/(3*nside**2)
            longitude=(np.arange(1,4*nr+1)-.5)*np.pi/(2*nr)
        elif ring<=3*nside:
            z=(2*nside-ring)*2/(3*nside)
            offset=.5*(1+((ring+nside)&1))
            longitude=(np.arange(1,4*nside+1)-offset)*np.pi/(2*nside)
        else:
            nr=4*nside-ring;z=-1+nr**2/(3*nside**2)
            longitude=(np.arange(1,4*nr+1)-.5)*np.pi/(2*nr)
        theta.extend(np.full(len(longitude),np.arccos(z)));phi.extend(longitude)
    return np.asarray(theta),np.asarray(phi)

