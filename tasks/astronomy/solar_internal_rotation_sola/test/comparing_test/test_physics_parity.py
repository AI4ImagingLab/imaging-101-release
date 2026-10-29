from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from src.preprocessing import load_data,radial_target_widths,cubic_interpolate
from src.physics_model import RotationOperator,ConstrainedOperator,initial_target,gaussian_targets
from src.solvers import reconstruct,solve_projected
ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def case():
    data=load_data(ROOT/'data/raw_data.npz');op=RotationOperator(data);con=ConstrainedOperator(op)
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as z:fixture={k:z[k] for k in z.files}
    return data,op,con,fixture

def test_original_full_forward_adjoint_and_constraint(case):
    data,op,con,f=case
    assert len(op.blocks)==293 and op.nm==29617 and op.np==5151
    assert_allclose(op.forward(f['probe_grid']),f['probe_forward'],rtol=1e-11,atol=1e-10)
    assert_allclose(op.adjoint(f['probe_forward']),f['probe_adjoint'],rtol=1e-11,atol=1e-10)
    assert_allclose(np.r_[con.signed,con.tau,con.v],f['householder'],rtol=1e-10,atol=1e-12)
    assert_allclose(con.constraint_kernel,f['constraint_kernel'],rtol=1e-10,atol=1e-12)
    assert_allclose(op.sqrt_weight*initial_target(data['mesh'])-con.constraint_kernel,f['start_vector'],rtol=1e-10,atol=1e-12)

def test_original_target_scaling_and_transformation(case):
    data,op,con,f=case;width=radial_target_widths(data)
    assert_allclose(width,f['target_widths'],rtol=1e-10,atol=1e-12)
    target=gaussian_targets(data['mesh'],width,f['target_indices'])
    rhs=op.sqrt_weight[:,None]*target-con.constraint_kernel[:,None]
    assert_allclose(f['U_probe'].T@rhs,f['transformed_targets'][f['basis_indices']],rtol=1e-9,atol=1e-11)
    assert_allclose(op.weight@target,np.ones(target.shape[1]),rtol=1e-14,atol=1e-14)
    assert_allclose(f['target_metadata'][:,3],np.repeat(width,51),rtol=1e-10,atol=1e-12)

def test_direct_acquisition_rows_and_independent_adjoint(case):
    data,op,con,f=case;rng=np.random.default_rng(35)
    x=rng.normal(size=op.np);y=rng.normal(size=op.nm)
    predicted=op.forward(x)
    for j in [0,1,17,550,4000,12000,29616]:
        a=data['radial_index'][j];b=data['angular_index'][j]
        row=np.outer(op.f1[:,a],op.g1[:,b])+np.outer(op.f2[:,a],op.g2[:,b])
        expected=np.sum(row*x.reshape(101,51))*op.inv_error[j]
        assert_allclose(predicted[j],expected,rtol=1e-12,atol=1e-12)
    assert_allclose(np.dot(op.forward(x),y),np.dot(x,op.adjoint(y)),rtol=1e-12,atol=1e-12)
    z=rng.normal(size=op.nm-1)
    assert_allclose(np.dot(con.forward(z),x),np.dot(z,con.adjoint(x)),rtol=1e-12,atol=1e-12)

def test_householder_physics(case):
    data,op,con,f=case
    h=op.forward(np.ones(op.np));rotated=con.householder(h)
    assert_allclose(rotated[0],con.signed,rtol=1e-13)
    assert_allclose(rotated[1:],0,atol=1e-13)
    assert_allclose(np.dot(con.q,h),1,rtol=1e-14)
    assert_allclose(np.dot(con.q,con.q),con.alpha**2,rtol=1e-14)
    v=np.arange(op.nm,dtype=float)/op.nm
    assert_allclose(con.householder(con.householder(v)),v,rtol=1e-12,atol=1e-12)

def test_cubic_interpolation_independent_polynomial():
    x=np.linspace(0,1,27);p=lambda z:2*z**3-3*z**2+.5*z+2
    for z in [-.1,0,.03,.71,1.,1.1]:assert_allclose(cubic_interpolate(x,p(x),z),p(z),rtol=1e-12,atol=1e-12)

def test_projected_inverse_against_dense_objective():
    rng=np.random.default_rng(61);a=rng.uniform(.3,3,9);b=rng.uniform(.2,2,9);target=rng.normal(size=(10,3));lam=3.
    B=np.zeros((10,9));B[np.arange(9),np.arange(9)]=a;B[np.arange(1,10),np.arange(9)]=b
    result=solve_projected(a,b,target,lam)
    expected=np.linalg.solve(B.T@B+lam*np.eye(9),B.T@target)
    assert_allclose(result,expected,rtol=1e-13,atol=1e-13)
    assert_allclose(B.T@(B@result-target)+lam*result,0,atol=1e-13)

def test_full_native_pipeline_and_uncertainty(case,tmp_path):
    data,op,con,f=case;output,d=reconstruct(data,iterations=1000)
    with np.load(ROOT/'notebooks/reference_outputs/upstream.npz',allow_pickle=False) as z:
        assert_allclose(output['reconstruction'],z['reconstruction'],rtol=0,atol=.001)
        assert_allclose(d['formal_error'],z['formal_error'][0],rtol=0,atol=1e-5)
    assert_allclose(d['averaging_kernels'].reshape(-1,5151),f['averaging_kernel_probes'],rtol=1e-7,atol=1e-8)
    norm=d['estimator_coefficients']@op.forward(np.ones(op.np))
    assert_allclose(norm,np.ones(len(norm)),rtol=1e-10,atol=1e-10)
    assert_allclose(d['covariance'].diagonal(),d['formal_error'].ravel()[d['kernel_indices']]**2,rtol=1e-10,atol=1e-10)
