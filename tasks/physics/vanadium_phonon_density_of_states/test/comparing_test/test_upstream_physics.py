from pathlib import Path
import importlib.util
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_observations,preprocess
from src.physics_model import *
from src.solvers import *
ROOT=Path(__file__).resolve().parents[2]
F=np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)
RAW=load_observations(ROOT/'data/raw_data.npz')

def test_original_preprocessing_precision_mask():
    i,v=preprocess(*RAW)
    assert_allclose(i,F['processed_intensity'],rtol=1e-9,atol=1e-10,equal_nan=True)
    assert_allclose(v,F['processed_variance'],rtol=1e-9,atol=1e-10,equal_nan=True)
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        assert np.array_equal(np.isfinite(z['intensity']),z['measured_mask'])
        assert np.isnan(z['intensity']).any()

def test_original_spectral_physics_stages():
    i,_=preprocess(*RAW);q=RAW[2]
    for label in ['initial','final']:
        g=F[label+'_g']
        e,a=spectral_orders(DOS_E,g)
        assert_allclose(e,F[label+'_A_energy'],rtol=0,atol=0)
        assert_allclose(a,F[label+'_A'],rtol=1e-9,atol=1e-10)
        assert_allclose(gamma0(DOS_E,g),F[label+'_gamma0'],rtol=1e-9,atol=1e-10)
        assert_allclose(debye_waller(q,DOS_E,g),F[label+'_DW'],rtol=1e-9,atol=1e-10)
        for k,value in forward(g,q,np.isfinite(i)).items():
            assert_allclose(value,F[label+'_'+k],rtol=1e-9,atol=1e-10,equal_nan=True)

def test_all_original_updates_and_final_uncertainty():
    result=reconstruct(*RAW)
    assert len(result['history'])==6
    for j,g in enumerate(result['history']):
        assert_allclose(g,F[f'round_{j}_g'],rtol=1e-9,atol=1e-10)
    assert_allclose(result['reconstruction'],F['final_g'],rtol=1e-9,atol=1e-10)
    assert_allclose(result['variance'],F['final_variance'],rtol=1e-9,atol=1e-10)
    r=reconstruct(*RAW,max_iterations=100,tolerance=1e-10)
    source=np.load(ROOT/'notebooks/reference_outputs/tight_source.npz',allow_pickle=False)
    assert len(r['history'])==14
    assert_allclose(r['reconstruction'],source['reconstruction'],rtol=1e-9,atol=1e-10)

def test_detailed_balance_higher_order_mass_and_source_a1_bias():
    e,a=spectral_orders(DOS_E,F['final_g'])
    assert_allclose(a[:,:79][:,::-1],a[:,80:]*np.exp(-BETA*e[80:]),rtol=1e-9,atol=1e-13)
    assert_allclose(a[1:].sum(axis=1),1,rtol=0,atol=1e-13)
    # The default six-round density differs slightly from the tighter comparison.
    assert_allclose(a[0].sum(),F['final_A'][0].sum(),rtol=1e-12,atol=1e-13)
    assert .98<a[0].sum()<.99
    assert np.isclose(F['final_g'].sum(),1)

def test_independent_debye_endpoint_grid_refinement():
    defects=[]
    for h in [1.,.5,.25,.125]:
        e=np.arange(round(55/h)+1)*h
        g=e**2;g/=g.sum()*h
        _,a=spectral_orders(e,g,orders=1,step=h)
        defects.append(1-a[0].sum()*h)
    assert all(.48<defects[j+1]/defects[j]<.52 for j in range(3))

def test_independent_recoil_units_and_kinematics():
    q=np.array([1.,2.])
    ratio=debye_waller(q,DOS_E,F['final_g'])
    assert_allclose(ratio[1]/ratio[0],4,rtol=1e-14)
    assert 0<ratio[0]<.1
    assert not kinematic_valid(np.array([0.]),np.array([1.]))[0,0]
    assert kinematic_valid(np.array([1.]),np.array([0.]))[0,0]
    measured,_=preprocess(*RAW)
    assert not np.any(np.isfinite(measured)&~kinematic_valid(RAW[2],TARGET_E))

def test_readme_si_to_inverse_angstrom_conversion_and_triangle():
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    assert r'c=\frac{\sqrt{2m_n(10^{-3}q_e)}}{\hbar_n\,10^{10}}' in readme
    # Direct p=sqrt(2mE), k=p/hbar, and meters-to-angstroms conversion.
    mass=1.6749286e-27
    joules_per_meV=1e-3*1.60217733e-19
    c=np.sqrt(2*mass*joules_per_meV)/(1.054571628e-34*1e10)
    assert .6946<c<.6948
    lower=c*(np.sqrt(120.)-np.sqrt(110.))
    upper=c*(np.sqrt(120.)+np.sqrt(110.))
    result=kinematic_valid(np.array([lower*.999,lower*1.001,upper*.999,upper*1.001]),np.array([10.]))[:,0]
    assert np.array_equal(result,[False,True,True,False])
