"""Full AMARES constrained LM initialization followed by bounded TRR fitting.
The acquisition, constraint expansion, residual and bound transforms are local.
Only general SciPy optimization is used; no MRS/domain package is imported.
"""
import numpy as np
from scipy.optimize import leastsq,least_squares
from .physics_model import forward
from .preprocessing import time_axis,interleave

PEAK_NAMES=['BATP', 'BATP2', 'BATP3', 'AATP', 'AATP2', 'GATP', 'GATP2', 'UDPG', 'NAD', 'PCr', 'GPC', 'GPE', 'Pin', 'Pex', 'PC', 'PE']
INITIAL_VALUES=np.array([1.409999966621399, -1938.0, 182.58937072753906, 0.0, 0.0, 0.7049999833106995, -1953.0, 182.58937072753906, 0.0, 0.0, 0.7049999833106995, -1923.0, 182.58937072753906, 0.0, 0.0, 1.5449999570846558, -898.7999877929688, 101.41060638427734, 0.0, 0.0, 1.5449999570846558, -914.7999877929688, 101.41060638427734, 0.0, 0.0, 1.5, -295.20001220703125, 122.58495330810547, 0.0, 0.0, 1.5, -311.20001220703125, 122.58495330810547, 0.0, 0.0, 0.07999999821186066, -1166.4000244140625, 101.69335174560547, 0.0, 0.0, 0.4099999964237213, -990.0, 127.20309448242188, 0.0, 0.0, 4.369999885559082, 0.0, 48.41194534301758, 0.0, 0.0, 1.3200000524520874, 354.0, 62.7061882019043, 0.0, 0.0, 0.800000011920929, 420.0, 60.00442123413086, 0.0, 0.0, 0.8500000238418579, 578.4000244140625, 66.09911346435547, 0.0, 0.0, 0.30000001192092896, 628.7999877929688, 97.10662841796875, 0.0, 0.0, 0.30000001192092896, 748.7999877929688, 62.7061882019043, 0.0, 0.0, 2.2699999809265137, 811.2000122070312, 71.09423828125, 0.0, 0.0],dtype=np.float64)
LOWER_VALUES=np.array([0.0, -1956.0, 170.6984368328014, -3.141592653589793, 0.0, 0.0, -1956.0, 170.6984368328014, -3.141592653589793, 0.0, 0.0, -1956.0, 170.6984368328014, -3.141592653589793, 0.0, 0.0, -926.4, 98.09937220099488, -3.141592653589793, 0.0, 0.0, -926.4, 98.09937220099488, -3.141592653589793, 0.0, 0.0, -318.0, 115.89963617623465, -3.141592653589793, 0.0, 0.0, -318.0, 115.89963617623465, -3.141592653589793, 0.0, 0.0, -1171.2, 91.49888603580273, -3.141592653589793, 0.0, 0.0, -990.0, 109.39968097595737, -3.141592653589793, 0.0, 0.0, -60.0, 41.50043895392117, -3.141592653589793, 0.0, 0.0, 352.8, 58.29853487266578, -3.141592653589793, 0.0, 0.0, 418.8, 54.50034935447573, -3.141592653589793, 0.0, 0.0, 577.1999999999999, 60.799242624923274, -3.141592653589793, 0.0, 0.0, 622.8000000000001, 69.40092331045213, -3.141592653589793, 0.0, 0.0, 746.4, 49.49893384996078, -3.141592653589793, 0.0, 0.0, 805.2, 64.89902103785795, -3.141592653589793, 0.0],dtype=np.float64)
UPPER_VALUES=np.array([np.inf, -1920.0, 194.4991427763977, 3.141592653589793, 1.0, np.inf, -1920.0, 194.4991427763977, 3.141592653589793, 1.0, np.inf, -1920.0, 194.4991427763977, 3.141592653589793, 1.0, np.inf, -890.4, 104.69985836618703, 3.141592653589793, 1.0, np.inf, -890.4, 104.69985836618703, 3.141592653589793, 1.0, np.inf, -286.8, 129.2985288437951, 3.141592653589793, 1.0, np.inf, -286.8, 129.2985288437951, 3.141592653589793, 1.0, np.inf, -1161.6, 111.90038872821484, 3.141592653589793, 1.0, np.inf, -980.4, 145.0002089264369, 3.141592653589793, 1.0, np.inf, 60.0, 55.30145548114113, 3.141592653589793, 1.0, np.inf, 355.2, 67.1012774880244, 3.141592653589793, 1.0, np.inf, 421.2, 65.4990652346936, 3.141592653589793, 1.0, np.inf, 579.6, 71.39897623813523, 3.141592653589793, 1.0, np.inf, 634.8, 124.79976816385454, 3.141592653589793, 1.0, np.inf, 751.1999999999999, 75.9008785107294, 3.141592653589793, 1.0, np.inf, 817.1999999999999, 77.29888724157686, 3.141592653589793, 1.0],dtype=np.float64)
FREE_INDICES=np.array([0, 1, 2, 3, 15, 16, 17, 25, 26, 27, 35, 36, 37, 40, 41, 42, 45, 46, 47, 50, 51, 52, 55, 56, 57, 60, 61, 62, 65, 66, 67, 70, 71, 72, 75, 76, 77],dtype=np.int64)

GROUP=np.array([0,0,0,3,3,5,5,7,8,9,10,11,12,13,14,15])
AMPLITUDE_RATIO=np.array([1,.5,.5,1,1,1,1,1,1,1,1,1,1,1,1,1])
FREQUENCY_OFFSET_HZ=np.array([0,-15,15,0,-16,0,-16,0,0,0,0,0,0,0,0,0])

def expand_parameters(free_values):
    """Expand 37 physical free parameters to all 80 original parameters.
    Multiplet amplitudes, chemical-shift separations, damping equality and the
    common receiver phase are explicit. Source bound clipping is preserved.
    """
    p=INITIAL_VALUES.copy();p[FREE_INDICES]=free_values;p=p.reshape(16,5)
    lo=LOWER_VALUES.reshape(16,5);hi=UPPER_VALUES.reshape(16,5)
    for k,base in enumerate(GROUP):
        if k!=base:
            p[k,0]=np.clip(p[base,0]*AMPLITUDE_RATIO[k],lo[k,0],hi[k,0])
            p[k,1]=np.clip(p[base,1]+FREQUENCY_OFFSET_HZ[k],lo[k,1],hi[k,1])
            p[k,2]=np.clip(p[base,2],lo[k,2],hi[k,2])
        p[k,3]=np.clip(p[0,3],lo[k,3],hi[k,3])
    return p

def internal_values(external):
    """Explicit Minuit bound transform used in original LM initialization."""
    lo=LOWER_VALUES[FREE_INDICES];hi=UPPER_VALUES[FREE_INDICES]
    out=np.empty_like(external)
    for k,(value,l,u) in enumerate(zip(external,lo,hi)):
        if not np.isfinite(l) and not np.isfinite(u):out[k]=value
        elif not np.isfinite(u):out[k]=np.sqrt((value-l+1.)**2-1.)
        elif not np.isfinite(l):out[k]=np.sqrt((u-value+1.)**2-1.)
        else:out[k]=np.arcsin(2*(value-l)/(u-l)-1)
        if abs(out[k])<1e-15:out[k]=0.
    return out

def external_values(internal):
    """Inverse bound map; output in original physical parameter units."""
    lo=LOWER_VALUES[FREE_INDICES];hi=UPPER_VALUES[FREE_INDICES]
    out=np.empty_like(internal)
    for k,(value,l,u) in enumerate(zip(internal,lo,hi)):
        if not np.isfinite(l) and not np.isfinite(u):out[k]=value
        elif not np.isfinite(u):out[k]=l-1.+np.sqrt(value*value+1.)
        elif not np.isfinite(l):out[k]=u+1.-np.sqrt(value*value+1.)
        else:out[k]=l+(np.sin(value)+1.)*(u-l)/2.
    return out

def residual(free_values,time,fid):
    """Measured minus modeled interleaved complex residual, no normalization."""
    return interleave(fid)-interleave(forward(expand_parameters(free_values),time))

def reconstruct(fid, use_lm=True, phase_constraint=True):
    """Fit all 16 resonance components using the full 1024-point FID.
    Returns the four required physical arrays plus identified stage diagnostics.
    The default retains the originalLMinitializer and uses the originalAPI's
    explicit TRRprecision controls to converge the physical least-squares target.
    use_lm=False is the independently initialized notebook TRR-only comparison.
    """
    if not phase_constraint:
        raise ValueError('This binding has a physically common receiver phase')
    fid=np.asarray(fid,dtype=np.complex128);time=time_axis(fid.size)
    tol=np.sqrt(np.abs(np.max(fid)))*1e-6
    start=INITIAL_VALUES[FREE_INDICES].copy()
    if use_lm:
        result=leastsq(lambda v:residual(external_values(v),time,fid),internal_values(start),
            full_output=1,col_deriv=0,ftol=tol,xtol=tol,gtol=0.,maxfev=2000,
            epsfcn=1e-10,factor=100,diag=None)
        if result[4] not in [1,2,3,4]:raise RuntimeError('LM failed: '+result[3])
        first=external_values(result[0]);lm_evaluations=result[2]['nfev']
    else:first=start;lm_evaluations=0
    final=least_squares(lambda v:residual(v,time,fid),first,
        bounds=(LOWER_VALUES[FREE_INDICES],UPPER_VALUES[FREE_INDICES]),
        jac='2-point',method='trf',ftol=1e-14,xtol=1e-14,gtol=1e-14,
        x_scale='jac',loss='linear',f_scale=1.,diff_step=None,max_nfev=4000)
    if not final.success:raise RuntimeError(final.message)
    p=expand_parameters(final.x)
    return dict(reconstruction=p[:,0],frequency_hz=p[:,1],linewidth_hz=p[:,2]/np.pi,
        phase_rad=p[:1,3],parameters=p,first_parameters=expand_parameters(first),
        fitted_fid=forward(p,time),residual=residual(final.x,time,fid),
        free_values=final.x,tolerance=tol,lm_evaluations=lm_evaluations,
        trr_evaluations=final.nfev,objective=float(2*final.cost))
