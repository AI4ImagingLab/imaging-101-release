"""Explicit angular light-transport basis and physical-field parameterization.

Five-basis mathematics adapted from Maeda/Hiura PolarimetricInterreflection,
revision b3e20b61754568b4db9255abb479c52494f4b6f4, MIT; see LICENSE.
"""
import numpy as np

def design_matrix(polarizer_psg,polarizer_psa,complete=False):
    """Rows are measured states;5 native or9 unrestricted linear Mueller terms."""
    g=np.asarray(polarizer_psg,dtype=np.float64);c=np.asarray(polarizer_psa,dtype=np.float64)
    if g.ndim!=1 or c.shape!=g.shape or not np.isfinite(g).all() or not np.isfinite(c).all():raise ValueError('Equal finite angle vectors required')
    cg,sg,cc,sc=np.cos(2*g),np.sin(2*g),np.cos(2*c),np.sin(2*c)
    terms=[np.ones_like(g),cc*cg,sc*sg,cc*sg,sc*cg]
    if complete:terms=[np.ones_like(g),cc,sc,cg,sg,cc*cg,sc*sg,cc*sg,sc*cg]
    return np.array(terms).T

def coefficients_to_fields(coefficients):
    """Native independent component plus full signed sum/difference quadratures."""
    x1,x2,x3,x4,x5=np.asarray(coefficients)
    sr,si=x2-x3,x4+x5;dr,di=x2+x3,-x4+x5
    forward=np.sqrt(sr**2+si**2);reverse=np.sqrt(dr**2+di**2)
    unpolarized=2*x1-(forward+reverse)
    return np.stack([unpolarized,sr,si,dr,di])

def fields_to_coefficients(fields):
    """Exact5-to5 inverse without fitting, clipping, phase wrapping or alignment."""
    u,sr,si,dr,di=np.asarray(fields)
    return np.stack([.5*(u+np.sqrt(sr*sr+si*si)+np.sqrt(dr*dr+di*di)),.5*(sr+dr),.5*(dr-sr),.5*(si-di),.5*(si+di)])

def forward(fields,polarizer_psg,polarizer_psa):
    """Instrumental HDR intensity, shape(frame,row,column), using all five fields."""
    return np.tensordot(design_matrix(polarizer_psg,polarizer_psa),fields_to_coefficients(fields),axes=(1,0))

def components(fields):
    """Return native amplitudes and half-angle phases, without calling a path truth."""
    u,sr,si,dr,di=np.asarray(fields)
    return {'unpolarized':u,'sum_amplitude':np.sqrt(sr*sr+si*si),'difference_amplitude':np.sqrt(dr*dr+di*di),
            'sum_phase':.5*np.arctan2(si,sr),'difference_phase':.5*np.arctan2(di,dr)}

def relative_residual(fields,intensity,polarizer_psg,polarizer_psa,block_rows=64):
    """Full-spatial residual accumulated in float64 without a hidden target mask."""
    W=design_matrix(polarizer_psg,polarizer_psa);X=fields_to_coefficients(fields)
    squared=0.;norm=0.;rms=np.empty(intensity.shape[1:]);frame_error=np.zeros(len(W));frame_norm=np.zeros(len(W))
    for start in range(0,intensity.shape[1],block_rows):
        stop=start+block_rows;y=np.clip(intensity[:,start:stop],0,None).astype(np.float64)
        difference=np.tensordot(W,X[:,start:stop],axes=(1,0))-y
        sums=(difference*difference).sum(axis=(1,2));ys=(y*y).sum(axis=(1,2));frame_error+=sums;frame_norm+=ys
        squared+=sums.sum();norm+=ys.sum();rms[start:stop]=np.sqrt(np.mean(difference*difference,axis=0))
    if norm<=0:raise ValueError('Zero-norm observations')
    return {'relative_l2':float(np.sqrt(squared/norm)),'rms_map':rms,'frame_relative_l2':np.sqrt(frame_error/frame_norm)}
