from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
from src.preprocessing import load_observations,prepare
from src.solvers import reconstruct,reconstruct_phase
from src.physics_model import wavelength_nm,frequency_filters,periodic_gradient,phase_to_induction,propagate,defocus_derivative
ROOT=Path(__file__).resolve().parents[2]
@pytest.fixture(scope='module')
def data():
    return load_observations(ROOT/'data/raw_data.npz')
@pytest.mark.parametrize('index,qc,symmetric,prefix',[(0,0.,False,''),(1,0.,False,'index1_'),(2,0.,False,'index2_'),(0,.0002,False,'qc0.0002_'),(0,.0005,False,'qc0.0005_'),(0,.001,False,'qc0.001_'),(0,0.,True,'symmetric_')])
def test_frozen_original_pipeline(data,index,qc,symmetric,prefix):
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        result=reconstruct(*data,index=index,qc=qc,symmetric=symmetric)
        for key in ['phase_B','phase_E','Bx','By']:
            np.testing.assert_allclose(result[key],z[prefix+key],rtol=2e-10,atol=2e-9)
def test_original_preprocessing_and_filter(data):
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        stack,i0,db,de=prepare(*data)
        np.testing.assert_array_equal(data[0],z['input_stack'])
        np.testing.assert_array_equal(data[1],z['input_flipstack'])
        for actual,key in [(stack,'selected'),(i0,'infocus'),(db,'difference_B'),(de,'difference_E')]:np.testing.assert_array_equal(actual,z[key])
        np.testing.assert_array_equal(frequency_filters(i0.shape)[0],z['qi'])
        from src.preprocessing import PIXEL_NM
        expected=-PIXEL_NM**2/(16*np.pi**3*wavelength_nm()*199065.)
        np.testing.assert_allclose(expected,z['prefactor'],rtol=1e-15)
def test_independent_coherent_propagation_sign_and_scale():
    y,x=np.indices((64,64));phase=.02*np.cos(2*np.pi*x/64);df=10000.
    difference=propagate(phase,df,pixel_nm=5)-propagate(phase,-df,pixel_nm=5)
    recovered=reconstruct_phase(np.ones(phase.shape),difference,df,pixel_nm=5)
    transfer=(np.sin(2*np.pi/64)/(2*np.pi/64))**2
    np.testing.assert_allclose(recovered-recovered.mean(),phase*transfer,atol=3e-9,rtol=1e-6)
    np.testing.assert_allclose(difference/(2*df),defocus_derivative(phase,np.ones(phase.shape),pixel_nm=5),atol=1e-11,rtol=2e-6)
def test_independent_induction_units_and_orientation():
    from scipy.constants import hbar,e
    y,x=np.indices((17,19));bx,by=phase_to_induction(3*y+2*x,pixel_nm=5)
    np.testing.assert_allclose(bx,3*hbar/e/5*1e18,rtol=1e-14)
    np.testing.assert_allclose(by,-2*hbar/e/5*1e18,rtol=1e-14)
def test_uniform_phase_gauge_and_divergence():
    y,x=np.indices((31,37));p=np.sin(x/7)*np.cos(y/5)
    bx,by=phase_to_induction(p);bx2,by2=phase_to_induction(p+7)
    np.testing.assert_allclose(bx,bx2,atol=1e-12);np.testing.assert_allclose(by,by2,atol=1e-12)
    div=np.gradient(bx,axis=1,edge_order=2)+np.gradient(by,axis=0,edge_order=2)
    np.testing.assert_allclose(div,0,atol=1e-12)
def test_flip_reversal_swaps_magnetic_sign_only(data):
    a=reconstruct(*data);b=reconstruct(data[1],data[0])
    np.testing.assert_allclose(a['phase_E'],b['phase_E'],atol=1e-10)
    np.testing.assert_allclose(a['phase_B']+b['phase_B'],a['phase_B'].max(),atol=1e-10)
    np.testing.assert_allclose(a['Bx'],-b['Bx'],atol=1e-10)
def test_periodic_gradient_constant_and_adjoint():
    rng=np.random.default_rng(12);a=rng.normal(size=(19,23));b=rng.normal(size=a.shape)
    for k in [0,1]:
        np.testing.assert_allclose(np.vdot(periodic_gradient(a)[k],b),-np.vdot(a,periodic_gradient(b)[k]),atol=1e-12)
    assert all(not v.any() for v in periodic_gradient(np.ones((17,17))))
def test_relativistic_wavelength():
    from scipy.constants import h,e,m_e,c
    kinetic=e*200000.
    momentum=np.sqrt(kinetic**2+2*kinetic*m_e*c*c)/c
    np.testing.assert_allclose(wavelength_nm(),h/momentum*1e9,rtol=1e-15)
def test_invalid_parameters(data):
    for q in [-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):reconstruct_phase(np.ones((8,8)),np.zeros((8,8)),1,qc=q)
    for index in [-1,3,True,1.2]:
        with pytest.raises(ValueError):prepare(*data,index=index)
