"""Explicit finite-exposure Doppler projection, background and floating default."""
import numpy as np
from scipy import sparse,ndimage
from .preprocessing import REST_WAVELENGTH,SOURCE_TWO_PI

def triangular_cdf(t):
    return np.where(t<0,0.,np.where(t<.5,2*t*t,np.where(t<1,1-2*(1-t)**2,1.)))

class DopplerModel:
    def __init__(self,prepared,nv=144):
        self.phase=np.asarray(prepared['phase_radians']);self.width=np.asarray(prepared['phase_width_radians']);self.velocity=np.asarray(prepared['velocity_cm_s']);self.nph=len(self.phase);self.nvp=len(self.velocity);self.nv=nv
        self.va=max(abs(self.velocity[0]),self.velocity[-1]);self.dv=2*self.va/(nv-1);self.denu=(self.velocity[-1]-self.velocity[0])/(self.nvp-1)/REST_WAVELENGTH*1e8
        coords=np.linspace(-self.va,self.va,nv);vx,vy=np.meshgrid(coords,coords,indexing='ij');support=vx*vx+vy*vy<(self.va+self.dv/10)**2
        self.ix,self.iy=np.where(support);self.vx=vx[support];self.vy=vy[support];self.n_pixels=len(self.ix)
        edges=np.r_[self.velocity[0]-(self.velocity[1]-self.velocity[0])/2,(self.velocity[:-1]+self.velocity[1:])/2,self.velocity[-1]+(self.velocity[-1]-self.velocity[-2])/2]
        rowparts=[];colparts=[];weightparts=[]
        for i,(phase,width) in enumerate(zip(self.phase,self.width)):
            v0=self.vx*np.sin(phase-width/2)-self.vy*np.cos(phase-width/2)
            v1=self.vx*np.sin(phase+width/2)-self.vy*np.cos(phase+width/2)
            low=np.minimum(v0,v1)-self.dv/2;high=np.maximum(v0,v1)+self.dv/2
            begin=np.clip(np.searchsorted(edges,low,side='left')-1,0,self.nvp-1)
            end=np.clip(np.searchsorted(edges,high,side='left')-1,0,self.nvp-1)
            for j in range(int(np.max(end-begin))+1):
                cols=np.flatnonzero(begin+j<=end);bins=begin[cols]+j
                a=(edges[bins]-low[cols])/(high[cols]-low[cols]);b=(edges[bins+1]-low[cols])/(high[cols]-low[cols])
                weights=(triangular_cdf(b)-triangular_cdf(a)).astype(np.float32)
                rowparts.append((i+bins*self.nph).astype(np.int32));colparts.append(cols.astype(np.int32));weightparts.append(weights)
        rows=np.concatenate(rowparts);cols=np.concatenate(colparts);weights=np.concatenate(weightparts)
        # Accumulation order is phase then velocity within each original column.
        totals=np.bincount(cols,weights=weights.astype(np.float64),minlength=self.n_pixels)
        weights=(weights.astype(np.float64)/totals[cols]).astype(np.float32)
        order=np.lexsort((rows//self.nph,rows%self.nph,cols))
        counts=np.bincount(cols,minlength=self.n_pixels);indptr=np.r_[0,np.cumsum(counts)]
        self.P_csc=sparse.csc_matrix((weights[order].astype(np.float64),rows[order],indptr),shape=(self.nph*self.nvp,self.n_pixels))
        self.P_csr=self.P_csc.tocsr()

    def forward(self,field):return self.P_csr@np.asarray(field)
    def backward(self,data):return self.P_csc.T@np.asarray(data)
    def scatter(self,vector):
        image=np.zeros((self.nv,self.nv));image[self.iy,self.ix]=vector;return image
    def smear(self,field,ns=7):
        width=int(1.5*ns);axis=np.arange(-width,width+1,dtype=np.float32)
        kernel=np.exp(-axis*axis/np.float32(ns*ns)).astype(np.float64);kernel/=sum(kernel)
        folded=self.scatter(field).T
        smooth=ndimage.convolve1d(folded,kernel,axis=1,mode='reflect')
        smooth=ndimage.convolve1d(smooth,kernel,axis=0,mode='reflect')
        return smooth[self.ix,self.iy]
    def initialize_data(self,spectra):
        observed=np.asarray(spectra).T.ravel().copy()
        baseline_field=((self.vx**2+self.vy**2)<self.va**2).astype(float)
        response=self.forward(baseline_field);central=response[(self.nvp//2)*self.nph-1]
        baseline=float(np.float32(.1))*sum(observed)/len(observed)
        baseline_data=baseline*response/central;baseline_field*=baseline/central
        data=observed+baseline_data;data[data<baseline/3]=float(np.float32(1e-5))*baseline
        # The source reduces the blanking interval around0 in cycle coordinates.
        phase_cycle=self.phase/SOURCE_TWO_PI
        reduced=phase_cycle-np.floor(phase_cycle)
        blank=(reduced>.95)|(reduced<.05)
        folded=data.reshape(self.nvp,self.nph)
        flux=np.sum(folded,axis=0);flux[blank]=0
        mean=sum(flux)/np.count_nonzero(~blank);relative=flux/mean
        # The original fixed-form "then wl=" assigns an unused THENWL variable.
        # Actual WL stays0: preserve its stated constant-integrated-flux behavior.
        wl=0.
        factors=np.ones(self.nph);valid=relative!=0;factors[valid]=(1+wl)/(relative[valid]+wl)
        folded*=factors
        weights=np.ones((self.nvp,self.nph));weights[response.reshape(self.nvp,self.nph)==0]=0;weights[:,blank]=0
        weights=weights.ravel();weights=weights/sum(weights)*self.nph
        total=sum(data)
        return dict(positive_data=data,baseline_data=baseline_data,baseline_field=baseline_field,weights=weights,total_flux=total,initial_field=np.full(self.n_pixels,total/self.n_pixels),blank_phases=blank)
