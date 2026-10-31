"""Explicit demodulation, steered receive aperture, DAS and phase velocimetry.

Extracted from official MUST, copyright2020 Damien Garcia,
LGPL-3.0-or-later. General FFT/filter/sparse primitives implement the formulas;
no MUST or ultrasound-domain package is imported.
"""
import numpy as np
from scipy import sparse,signal,optimize,ndimage

SOUND_SPEED=1480.
CARRIER_FREQUENCY=5e6
SAMPLING_FREQUENCY=6666666.666666667
PULSE_REPETITION_FREQUENCY=10000.
START_TIME=9.95e-6
PITCH=.000298
ELEMENT_WIDTH=.000262
BANDWIDTH_PERCENT=15.
RECEIVE_ANGLE=np.pi/12

def demodulate(rf,sampling_frequency=SAMPLING_FREQUENCY):
    """Complex downmix and fifth-order zero-phase low-pass, time axis first.

    Normalized cutoff is bandwidthHz/Fs. Octave's independent-trace filtfilt
    uses three filter orders of odd-reflected padding, not SciPy's default
    three coefficient counts. The factor two preserves envelope amplitude.
    """
    rf=np.asarray(rf,dtype=float)
    if rf.ndim!=3 or rf.shape[0]<=15 or not np.isfinite(rf).all():raise ValueError('RF must have finite time/element/ensemble axes')
    time=START_TIME+np.arange(rf.shape[0],dtype=float)/sampling_frequency
    mixed=rf*np.exp(-1j*2*np.pi*CARRIER_FREQUENCY*time)[:,None,None]
    b,a=signal.butter(5,CARRIER_FREQUENCY*BANDWIDTH_PERCENT/100/sampling_frequency)
    return 2*signal.filtfilt(b,a,mixed,axis=0,padtype='odd',padlen=15)

def automatic_fnumber(receive_angle):
    """Known element-directivity criterion, independently of any target motion."""
    rx=abs(float(receive_angle))
    wavelength=SOUND_SPEED/(CARRIER_FREQUENCY*(1+BANDWIDTH_PERCENT/200))
    objective=lambda alpha:abs(np.cos(alpha+rx)*np.sinc(ELEMENT_WIDTH/wavelength*np.sin(alpha+rx))-.71)
    answer=optimize.minimize_scalar(objective,bounds=(0,np.pi/2-rx),method='bounded',options={'xatol':np.pi/100})
    return float(.5/np.tan(answer.x))

def beamformer(x,z,receive_angle,n_fast=334,sampling_frequency=SAMPLING_FREQUENCY,chunk_size=2048):
    """Explicit complex sparse linear-interpolation receive operator.

    Grid arrays have matching shapes, flattened in Fortran order. Columns
    have fast-time first, then128 elements. All measured TX delays are zero.
    Preserve the source virtual-array parabolic TX-distance reader instead
    of silently replacing it by the ideal plane-wave distance z.
    """
    x=np.asarray(x,dtype=float).ravel(order='F');z=np.asarray(z,dtype=float).ravel(order='F')
    if x.shape!=z.shape or not np.isfinite(x).all() or not np.isfinite(z).all() or np.any(z<=0):raise ValueError('Invalid beamforming coordinates')
    elements=(np.arange(128)-(128-1)/2)*PITCH
    fnumber=automatic_fnumber(receive_angle)
    all_rows=[];all_cols=[];all_values=[]
    for first in range(0,len(x),chunk_size):
        xx=x[first:first+chunk_size,None];zz=z[first:first+chunk_size,None]
        dx=xx-elements[None,:]
        distance=np.hypot(dx,zz)
        normal_distance=abs(dx)
        closest=np.argmin(normal_distance,axis=1)
        row_index=np.arange(len(xx));dtx=distance[row_index,closest].copy()
        interior=(closest>0)&(closest<127);r=row_index[interior];j=closest[interior]
        left=distance[r,j-1];right=distance[r,j+1]
        dnleft=normal_distance[r,j-1];dnmid=normal_distance[r,j];dnright=normal_distance[r,j+1]
        dtx[interior]=(left+right)/2-(right-left)*(dnright-dnleft)/(2*dnleft-4*dnmid+2*dnright)
        travel=(dtx[:,None]+distance)/SOUND_SPEED
        index=(travel-START_TIME)*sampling_frequency+1
        valid=(index>=1)&(index<=n_fast-1)
        valid &= abs(dx-zz*np.tan(receive_angle))<=zz/np.cos(receive_angle)/2/fnumber
        row,element=np.nonzero(valid)
        at=index[row,element];lower=np.floor(at).astype(np.int64)
        fraction=at-lower
        phase=np.exp(1j*2*np.pi*CARRIER_FREQUENCY*travel[row,element])
        columns=lower-1+element*n_fast
        all_rows.extend([row+first,row+first]);all_cols.extend([columns,columns+1])
        all_values.extend([(1-fraction)*phase,fraction*phase])
    matrix=sparse.coo_matrix((np.concatenate(all_values),(np.concatenate(all_rows),np.concatenate(all_cols))),shape=(len(x),n_fast*128)).tocsr()
    matrix.sort_indices()
    return matrix

def beamform(iq,x,z,receive_angle,sampling_frequency=SAMPLING_FREQUENCY):
    matrix=beamformer(x,z,receive_angle,n_fast=iq.shape[0],sampling_frequency=sampling_frequency)
    result=matrix@iq.reshape(iq.shape[0]*128,iq.shape[2],order='F')
    return result.reshape((*np.asarray(x).shape,iq.shape[2]),order='F')

def doppler_velocity(iq,window=11,lag=1,pulse_repetition_frequency=PULSE_REPETITION_FREQUENCY):
    """Full-ensemble lag correlation, replicate Hamming averaging and signed phase.

    Return radial velocity(m/s), original finite-packet variance estimator
    ((m/s)^2), and complex averaged autocorrelation. The variance is retained
    as the author's estimator, not relabeled as calibrated uncertainty.
    """
    iq=np.asarray(iq)
    if iq.ndim!=3 or not 0<lag<iq.shape[2] or window<1:raise ValueError('Invalid Doppler packet/lag/window')
    ac=np.sum(iq[:,:,:-lag]*np.conj(iq[:,:,lag:]),axis=2)
    power=np.sum(iq.real**2+iq.imag**2,axis=2)
    if window!=1:
        h=np.outer(np.hamming(window),np.hamming(window))
        ac=ndimage.convolve(ac.real,h,mode='nearest')+1j*ndimage.convolve(ac.imag,h,mode='nearest')
        power=ndimage.convolve(power,h,mode='nearest')
    nyquist=SOUND_SPEED*pulse_repetition_frequency/(4*CARRIER_FREQUENCY*lag)
    velocity=-nyquist*np.angle(ac)/np.pi
    variance=2*(nyquist/np.pi)**2*(1-abs(ac)/power)
    return velocity,variance,ac

def invert_receive_projections(radial_plus,radial_minus,receive_angle=RECEIVE_ANGLE):
    """The author's bistatic two-receive-angle signed vector formula."""
    return np.stack([(radial_minus-radial_plus)/np.sin(receive_angle),-(radial_minus+radial_plus)/(1+np.cos(receive_angle))])

def predicted_radial(velocity,receive_angle=RECEIVE_ANGLE):
    vx,vz=velocity;s=np.sin(receive_angle);c=1+np.cos(receive_angle)
    return np.stack([-.5*(s*vx+c*vz),.5*(s*vx-c*vz)])

def phase_consistency(velocity,autocorrelation,roi):
    phase=autocorrelation[:,roi];weight=abs(phase)
    nyquist=SOUND_SPEED*PULSE_REPETITION_FREQUENCY/(4*CARRIER_FREQUENCY)
    prediction=np.exp(-1j*np.pi*predicted_radial(velocity)[:,roi]/nyquist)
    return float(np.sqrt(np.sum(weight*abs(prediction-phase/weight)**2)/weight.sum()))
