import numpy as np
from src.preprocessing import reflecting_laplacian,fill_gaps
from src.physics_model import arc_beta,physical_kappa,fit_parabola
from src.physics_model import Spectrum

def test_biharmonic_constant_plane_boundary_and_adjoint():
    y,x=np.mgrid[:9,:10];plane=4.+2*x+y
    missing=np.zeros_like(plane,bool);missing[4,4]=True;missing[4,5]=True
    observed=plane.copy();observed[missing]=0
    np.testing.assert_allclose(fill_gaps(observed),plane,rtol=1e-13,atol=1e-13)
    matrix=reflecting_laplacian(plane.shape)
    np.testing.assert_array_equal(matrix.toarray(),matrix.T.toarray())
    np.testing.assert_allclose(matrix@np.ones(plane.size),0,atol=0)

def test_cyclic_fourier_axes_and_no_missing_2pi():
    d=Spectrum();nf,nt=128,64;d.dt=20.;d.df=.5;d.dlam=.002
    nr,nc=256,128;beta=5/(nr*d.dlam);doppler=7/(nc*d.dt)
    rows=np.arange(nf)*d.dlam;times=np.arange(nt)*d.dt
    d.lamdyn=np.cos(2*np.pi*(beta*rows[:,None]+doppler*times[None,:]))
    d.calc_sspec(lamsteps=True,window=None)
    peak=np.unravel_index(np.argmax(d.lamsspec),d.lamsspec.shape)
    np.testing.assert_allclose(d.beta[peak[0]],beta,rtol=1e-14)
    np.testing.assert_allclose(d.fdop[peak[1]]/1000,doppler,rtol=1e-14)
    kappa=2*beta/doppler**2
    np.testing.assert_allclose(arc_beta(doppler,kappa),beta)
    np.testing.assert_allclose(physical_kappa(beta/(doppler*1000)**2),kappa)

def test_corrected_parabola_uncertainty_has_physical_scale_covariance():
    x=np.linspace(2.,4.,100);y=-(x-3.15)**2+.01*np.sin(13*x)
    _,peak,error=fit_parabola(x,y)
    _,scaled_peak,scaled_error=fit_parabola(100*x,y)
    np.testing.assert_allclose(scaled_peak,100*peak,rtol=1e-12)
    np.testing.assert_allclose(scaled_error,100*error,rtol=1e-10)
    assert error>0 and abs(peak-3.15)<.01
    # Independent numerical differentiation of the peak as a function of all
    # three coefficients validates propagation, including cross covariance.
    scale=1000/np.ptp(x);coefficient,covariance=np.polyfit(x*scale,y,2,cov=True)
    numerical=[]
    for j in range(3):
        step=1e-5*max(abs(coefficient[j]),1e-10)
        plus=coefficient.copy();minus=coefficient.copy();plus[j]+=step;minus[j]-=step
        numerical.append((-plus[1]/(2*plus[0])/scale+minus[1]/(2*minus[0])/scale)/(2*step))
    numerical=np.array(numerical)
    np.testing.assert_allclose(error,np.sqrt(numerical@covariance@numerical),rtol=1e-6)

def test_source_mean_refill_is_explicit_and_keeps_observations():
    data=np.array([[0.,2.],[3.,5.]])
    filled=fill_gaps(data,'mean')
    assert filled[0,0]==10/3
    np.testing.assert_array_equal(filled[data!=0],data[data!=0])

def test_public_data_table_and_physical_acquisition_constants():
    from pathlib import Path
    import scipy.constants as constants
    root=Path(__file__).resolve().parents[2];text=(root/'README.md').read_text(encoding='utf8')
    with np.load(root/'data/raw_data.npz') as f:
        for key in f.files:
            assert f'| `{key}` | `{f[key].shape}` | `float64` |' in text
    assert f'| Speed of light c | {int(constants.c)} | m/s |' in text
    assert '| Seconds per day | 86400 | s/day |' in text
