import numpy as np
from src.physics_model import design_matrix,forward,adjoint,coefficients_to_tensor,tensor_to_coefficients
from src.preprocessing import load_data,window_waveforms,distance_weights,SHIFTS,NPTS,STATIONS
from src.solvers import solve,variance_reduction
from src.visualization import tensor_relative_error,radiation_pattern

def test_analytic_metrics():
    assert variance_reduction(np.ones((2,3,4)),np.zeros((2,3,4)),np.ones(2))==0
    assert variance_reduction(np.ones((2,3,4)),np.ones((2,3,4)),np.ones(2))==100
    assert tensor_relative_error(np.eye(3)*2,np.eye(3))==1
    assert np.max(np.abs(radiation_pattern(np.eye(3)).compressed()-1/np.sqrt(3)))<1e-14

def test_units_tensor_symmetry_and_linearity():
    a=np.arange(1.,7.);M=coefficients_to_tensor(a)
    assert np.array_equal(M,M.T)
    np.testing.assert_array_equal(tensor_to_coefficients(M),a)
    assert M[0,0]==1e13
    d=load_data();g=d['fundamental_greens'][0];az=d['station_azimuth']
    np.testing.assert_allclose(forward(g,az,3*M),3*forward(g,az,M),rtol=1e-14,atol=1e-18)

def test_adjoint_and_finite_difference_objective():
    d=load_data();G=design_matrix(d['fundamental_greens'][0],d['station_azimuth'])
    rng=np.random.default_rng(31);a=rng.normal(size=6);r=rng.normal(size=G.shape[0])
    np.testing.assert_allclose(np.dot(G@a,r),np.dot(a,adjoint(G,r)),rtol=1e-12,atol=1e-18)
    y=window_waveforms(d['waveforms']).ravel();w=np.repeat(distance_weights(d['station_distance']),450)
    grad=G.T@(w*(G@a-y));v=rng.normal(size=6);h=1e-4
    fun=lambda x:.5*np.sum(w*(G@x-y)**2)
    np.testing.assert_allclose((fun(a+h*v)-fun(a-h*v))/(2*h),grad@v,rtol=1e-7,atol=1e-20)

def test_isotropic_and_coordinate_symmetries():
    rng=np.random.default_rng(20);g=rng.normal(size=(2,10,9));az=np.array([11.,42.])
    pred=forward(g,az,np.eye(3)*1e13)
    np.testing.assert_array_equal(pred[:,2],0)
    np.testing.assert_allclose(pred[:,0],-.000004*g[:,2]+.99999*g[:,3],atol=2e-15)
    np.testing.assert_allclose(pred[:,1],-.000004*g[:,6]+.99999*g[:,7],atol=2e-15)
    G=design_matrix(g,az);G2=design_matrix(g,az+180)
    np.testing.assert_allclose(G2[:,[0,1,3,5]],G[:,[0,1,3,5]],atol=3e-15)
    np.testing.assert_allclose(G2[:,[2,4]],-G[:,[2,4]],atol=3e-15)

def test_normal_equation_stationarity_and_deviatoric_constraint():
    d=load_data();r=solve(d);G=design_matrix(d['fundamental_greens'][0],d['station_azimuth'])
    obs=window_waveforms(d['waveforms']).ravel();w=np.repeat(distance_weights(d['station_distance']),450)
    a=tensor_to_coefficients(r['reconstruction'][0]);res=G.T@(w*(G@a-obs))
    assert np.linalg.norm(res)/np.linalg.norm(G.T@(w*obs))<1e-12
    rr=solve(d,degree=5)
    assert np.max(np.abs(np.trace(rr['tensors'],axis1=1,axis2=2)))<10

def test_readme_constants_and_source_import_backend():
    from pathlib import Path
    text=Path('README.md').read_text(encoding='utf-8')
    assert '32,32,31,31,31,31,31,32' in text and '150' in text
    assert STATIONS[-1]=='BK.MNRC.00'
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before


def test_document_math_backslashes_and_control_characters():
    from pathlib import Path
    import json,re
    for file in [Path('README.md'),Path('plan/approach.md'),Path('plan/design.md')]:
        text=file.read_text(encoding='utf-8')
        assert not any(ord(c)<32 and c!='\n' for c in text)
    text=Path('README.md').read_text(encoding='utf-8')
    for command in ['sum','epsilon','qquad','frac','cos','sin','alpha','mathrm']:
        assert chr(92)+command in text
    notebook=json.loads(Path('notebooks/california_seismic_source_tensor.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        assert not any(ord(c)<32 and c!='\n' for c in ''.join(cell['source']))
