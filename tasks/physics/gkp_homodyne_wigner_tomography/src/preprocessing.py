"""Measured ADC/vacuum calibration, native phase fitting and PNR confidence selection.

Apache-2.0 adaptation of original2025 notebook cells11–17, preserving all samples.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

RAW_SAMPLE_COUNT=3756232
FILE_COUNT=100
PHASE_COUNT=64
CONFIDENCE_RATIO=.8
CONFIDENCE_HISTOGRAM_BINS=201

def load_observations(path=None):
    path=Path(path) if path is not None else Path(__file__).resolve().parents[1]/'data/raw_data.npz'
    shapes={'adc':(1,RAW_SAMPLE_COUNT),'pnr_distance':(1,RAW_SAMPLE_COUNT,3),
            'phase_index':(1,RAW_SAMPLE_COUNT),'file_index':(1,RAW_SAMPLE_COUNT),
            'quadrature_mean':(1,FILE_COUNT),'phase_step':(1,FILE_COUNT),
            'vacuum_variance':(1,FILE_COUNT,PHASE_COUNT),'signal_variance':(1,FILE_COUNT,PHASE_COUNT),
            'variance_phase':(1,FILE_COUNT,PHASE_COUNT)}
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(shapes):raise ValueError('Unexpected observation/calibration keys')
        result={}
        for k,shape in shapes.items():
            a=z[k]
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Invalid array '+k)
            result[k]=a[0].copy()
    if np.any(result['vacuum_variance']<=0):raise ValueError('Vacuum variance must be positive')
    if np.any(result['file_index']<0) or np.any(result['file_index']>=FILE_COUNT):raise ValueError('Invalid file calibration index')
    return result

def phase_distortion(x,offset,amplitude,phase):
    return x-offset+amplitude*np.sin(x-phase)

def variance_model(x,offset,amplitude,phase,squeezed,antisqueezed):
    angle=phase_distortion(x,offset,amplitude,phase)
    return squeezed*np.cos(angle)**2+antisqueezed*np.sin(angle)**2

def calibrate(data):
    """Return full calibrated samples and actual strictly selected conditional samples."""
    parameters=np.zeros((FILE_COUNT,5))
    for i in range(FILE_COUNT):
        svar=data['signal_variance'][i]/data['vacuum_variance'][i]*.5
        theta=data['variance_phase'][i]
        initial=[((theta[np.argmin(svar)]+np.pi/2)%np.pi)-np.pi/2,0,0,svar.min(),svar.max()]
        parameters[i],_=curve_fit(variance_model,theta,svar,p0=initial)
    q=[];phi=[];uncorrected=[]
    # Keep original per-file arithmetic and concatenation ordering exactly.
    for i in range(FILE_COUNT):
        mask=data['file_index']==i
        q.append((data['adc'][mask]-data['quadrature_mean'][i])/np.sqrt(2*np.mean(data['vacuum_variance'][i])))
        phase=data['phase_index'][mask]*data['phase_step'][i]
        uncorrected.append(phase)
        phi.append(phase_distortion(phase,*parameters[i,:3]))
    q=np.concatenate(q);phi=np.concatenate(phi);uncorrected=np.concatenate(uncorrected)
    distance=data['pnr_distance'].astype(np.float64)
    cutoff=np.zeros(3)
    for i in range(3):
        hist,edges=np.histogram(distance[:,i],bins=CONFIDENCE_HISTOGRAM_BINS)
        centres=(edges[1:]+edges[:-1])/2
        cutoff[i]=centres[np.argmin(np.abs(np.cumsum(hist/hist.sum())-CONFIDENCE_RATIO))]
    indices=np.where((distance[:,0]<cutoff[0])*(distance[:,1]<cutoff[1])*(distance[:,2]<cutoff[2]))[0]
    return {'q':q,'phi':phi,'uncorrected_phi':uncorrected,'phase_fit_param':parameters,
            'pnr_distance':distance,'cutoff':cutoff,'accepted_indices':indices,
            'q_accepted':q[indices],'phi_accepted':phi[indices],
            'uncorrected_phi_accepted':uncorrected[indices]}
