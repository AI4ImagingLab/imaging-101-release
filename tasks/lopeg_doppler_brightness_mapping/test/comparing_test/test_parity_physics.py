from pathlib import Path
import importlib.util,subprocess,sys
import numpy as np
import pytest
from src import preprocessing as pre,physics_model as phy,solvers as mem
ROOT=Path(__file__).resolve().parents[2]
MODE=pre.MODE
@pytest.fixture(scope='module')
def oracle():
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
@pytest.fixture(scope='module')
def run():return pre.reconstruct(pre.load_data(ROOT/'data/raw_data.npz'))
def test_exact_original_spatial_output(run,oracle):
    expected=oracle['output_brightness' if MODE=='brightness' else 'output_vector_field']
    np.testing.assert_allclose(run[0],expected,rtol=1e-9,atol=1e-8)
def test_original_full_iteration_trajectory(run,oracle):
    np.testing.assert_allclose(run[1].history,oracle['output_iteration_diagnostics'],rtol=1e-8,atol=1e-7)
def test_original_forward_and_response(run,oracle):
    np.testing.assert_allclose(run[1].first['model'],oracle['input_model'],rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose(run[1].first['response'],oracle['input_response'],rtol=1e-9,atol=1e-13)
    np.testing.assert_allclose(run[1].final_profile,oracle['output_final_profile'],rtol=1e-8,atol=1e-12)
def test_original_first_update(oracle):
    keys=['config_n1','config_n2','config_ntot','input_image','input_data','input_model','input_variance','input_response','param_weights','param_default_image','param_default_B','param_max_brightness','config_target','config_fix_entropy']
    args=[oracle[k].copy() for k in keys]
    for i in [0,1,2,13]:args[i]=int(args[i])
    result=mem.mem_iter(*args)
    for got,key in zip(result,['output_entropy','output_chi2','output_test','output_next_image','output_entropy_image','output_entropy_ff','output_entropy_B']):
        np.testing.assert_allclose(got,oracle[key],rtol=1e-9,atol=1e-8)
def test_original_geometry_and_line_calibration(run,oracle):
    s=run[1]
    for attr,key in [('clat','param_colatitude'),('long','param_longitude'),('radius','param_radius'),('area','param_area')]:
        np.testing.assert_allclose(getattr(s.grid,attr),oracle[key],rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(s.line.str,oracle['param_line_strength'],rtol=1e-13,atol=1e-14)
    for key in ['viewAngle','visible','projArea']:
        np.testing.assert_allclose(getattr(s.views[0],key),oracle['output_first_'+key],rtol=1e-13,atol=1e-14)
def test_legendre_analytic_values_and_derivative():
    for x in [-.83,-.2,.12,.91]:
        p,d=phy.legendre_values(3,x)
        assert p[0,1]==pytest.approx(x,abs=1e-14)
        assert p[1,1]==pytest.approx(-np.sqrt(1-x*x),abs=1e-14)
        assert d[1,1]==pytest.approx(x/np.sqrt(1-x*x),abs=1e-13)
        h=1e-6;plus=phy.legendre_values(3,x+h)[0];minus=phy.legendre_values(3,x-h)[0]
        np.testing.assert_allclose(d,(plus-minus)/(2*h),rtol=1e-7,atol=1e-7)
def test_sphere_geometry_area_and_normals():
    g=phy.starGrid(60,verbose=0)
    assert np.sum(g.area)==pytest.approx(4*np.pi,rel=1e-13)
    np.testing.assert_allclose(np.linalg.norm(g.getCartesianNormals(),axis=0),1,atol=1e-14)
    assert np.all(g.area>0)
def test_harmonic_cartesian_basis(run,oracle):
    s=run[1];basis=s.field.getAllMagDerivsCart()
    if MODE=='magnetic':np.testing.assert_allclose(basis,oracle['output_magnetic_basis'],rtol=1e-9,atol=1e-11)
    f=phy.magSphHarmoics(1);f.initMagGeom(s.grid.clat,s.grid.long);f.alpha[0]=1.
    np.testing.assert_allclose(f.getAllMagVectors()[0],np.sqrt(3/(4*np.pi))*np.cos(s.grid.clat),atol=1e-14)
    np.testing.assert_allclose(np.sum(f.getAllMagVectors()**2,axis=0),np.sum(f.getAllMagVectorsCart()**2,axis=0),rtol=1e-13,atol=1e-14)
def test_entropy_gradient_and_curvature(oracle):
    n=int(oracle['config_ntot']);n1=int(oracle['config_n1']);n2=int(oracle['config_n2'])
    image=oracle['input_image'].copy();image+=.03
    weights=oracle['param_weights'];args=(n1,n2,n,image,weights,1.,100.,1.01)
    s,gradient,curvature,*_=mem.get_s_grads(*args)
    for index in [0,n//2,n-1]:
        h=1e-5;plus=image.copy();minus=image.copy();plus[index]+=h;minus[index]-=h
        sp=mem.get_s_grads(n1,n2,n,plus,weights,1.,100.,1.01)
        sm=mem.get_s_grads(n1,n2,n,minus,weights,1.,100.,1.01)
        assert (sp[0]-sm[0])/(2*h)==pytest.approx(gradient[index],rel=1e-4,abs=1e-8)
        assert (sp[1][index]-sm[1][index])/(2*h)==pytest.approx(curvature[index],rel=1e-5,abs=1e-8)
def test_response_finite_difference(oracle):
    s=pre.prepare(pre.load_data(ROOT/'data/raw_data.npz'));p=s.par
    image=mem.packImageVector(s.bright,s.field,'full',p.fitBri,p.fitMag).copy()
    response=oracle['input_response']
    for index in [0,len(image)//3,len(image)-2]:
        h=1e-4;predictions=[]
        for sign in [1,-1]:
            trial=image.copy();trial[index]+=sign*h
            mem.unpackImageVector(trial,s.bright,s.field,'full',p.fitBri,p.fitMag)
            cart=s.field.getAllMagVectorsCart()
            for v,profile in zip(s.views,s.profiles):profile.updateIntProfDeriv(v,cart,0.,s.bright,s.line,0,0)
            predictions.append(mem.packModelVector(s.profiles,p.fitBri,p.fitMag))
        np.testing.assert_allclose((predictions[0]-predictions[1])/(2*h),response[:,index],rtol=2e-5,atol=2e-10)
def test_spatial_output_forward_closure(run):
    x,s=run;profile=pre.project_spatial(x,s)
    chi2=np.mean(((profile-s.observation.reshape(profile.shape))/np.sqrt(s.variance).reshape(profile.shape))**2)
    assert chi2<1.002
    if MODE=='magnetic':
        zero=pre.project_spatial(np.zeros_like(x),s);np.testing.assert_array_equal(zero,np.zeros_like(zero))
        doubled=pre.project_spatial(2*x,s);np.testing.assert_allclose(doubled,2*profile,atol=1e-16)
def test_main_import_preserves_backend():
    code="import matplotlib; before=matplotlib.get_backend(); import src.main; assert before==matplotlib.get_backend()"
    assert subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True).returncode==0
