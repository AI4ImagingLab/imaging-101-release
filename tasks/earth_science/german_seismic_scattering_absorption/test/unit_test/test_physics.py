from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import *
from src.physics_model import *
from src.solvers import weighted_solve

def test_energy_quadratic_and_noise_units():
    t=np.arange(4401)/20;v=np.stack([np.sin(2*np.pi*t),np.cos(2*np.pi*t),np.zeros_like(t)])*1e-7
    a=observed_energy(v,[.5,2]);b=observed_energy(3*v,[.5,2])
    np.testing.assert_allclose(b,9*a,rtol=1e-12,atol=1e-25)
    assert np.all(a>=0) and filter_width([.5,2])>0

def test_green_causality_direct_energy_and_scaling():
    distance=34000.;g0=1e-5;t=np.arange(0,30,.05);g=green(distance,t,g0);arrival=distance/VELOCITY;i=200
    assert np.all(g[:i]==0) and np.all(g[i:]>=0)
    np.testing.assert_allclose(g[i]*.05*4*np.pi*distance**2*VELOCITY,np.exp(-distance*g0),rtol=1e-12)
    # Spatial/time scaling with mean free path: energy density is inverse volume.
    c1=coda_green(distance,np.array([20.,30.]),g0)
    c2=coda_green(2*distance,np.array([40.,60.]),g0/2)
    np.testing.assert_allclose(c2,c1/8,rtol=1e-12)

def test_weighted_least_squares_known_solution_and_stationarity():
    a=np.array([[1.,0],[1,1],[1,2],[1,3]]);w=np.array([1.,2,3,4]);true=np.array([2.,-.3]);b=a@true
    coeff,error=weighted_solve(a,b,w);np.testing.assert_allclose(coeff,true,atol=1e-14);assert error<1e-14
    b=b+np.array([.1,-.2,.2,-.1]);coeff,error=weighted_solve(a,b,w)
    np.testing.assert_allclose(a.T@(w*(b-a@coeff)),0,atol=1e-13)

def test_sample_rounding_and_smoothing():
    assert sample_index(.025,0)==1 and sample_index(.024,0)==0
    np.testing.assert_allclose(smooth(np.ones(200))[20:-20],1,rtol=1e-15)

def test_visualization_metric_independent_known_case():
    from src.visualization import log_ratio_metrics
    mask=np.array([True,False]);reference={'reconstruction':np.ones((1,2)), 'site_amplification':np.array([[1.,0.]])}
    prediction={'reconstruction':np.array([[np.e,1.]]), 'site_amplification':np.array([[np.e**2,0.]])}
    metrics=log_ratio_metrics(prediction,reference,mask)
    np.testing.assert_allclose([metrics['reconstruction'],metrics['site_amplification']],[1.,2.],atol=1e-14)

def test_visible_constants_and_measurement_identity():
    d=load_data(ROOT/'data/raw_data.npz');assert d['valid'].sum()==24 and d['fit_valid'].sum()==119
    assert d['velocity'].shape==(5,5,3,4401) and np.all(d['velocity'][~d['valid']]==0)
    # ObsPy timestamps preserve microsecond precision; distance is continuous.
    np.testing.assert_array_equal(d['s_onset'][d['valid']],np.round(d['distance'][d['valid']]/3400,6))
    assert SAMPLE_RATE==20 and DENSITY==2700 and FREE_SURFACE==4 and VELOCITY==3400

def test_readme_acquisition_constants():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for value in ['| Sampling rate | 20 | Hz |','| Medium density | 2700 | kg m^-3 |','| Propagation speed | 3400 | m s^-1 |','| Free-surface energy factor | 4 | dimensionless |']:
        assert value in text
