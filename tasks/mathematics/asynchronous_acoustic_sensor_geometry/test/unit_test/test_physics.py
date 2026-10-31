import numpy as np
from src.physics_model import pack,unpack,residual_jacobian,predict_measurements
from src.solvers import normal_step
from src.preprocessing import load_data
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_independent_time_of_flight_and_clock_signs():
    # Two sensors, three events: independently derive delays from collinear
    # sensor/source distances, with distinct offsets and clock scales.
    state=np.zeros((5,5));state[:2,:3]=[[0,2,0],[3,0,0]]
    state[1,3]=.002;state[:2,4]=[1e-5,-2e-5]
    state[2:,:3]=[[0,0,0],[1,0,0],[0,1,0]]
    intervals=np.array([0.,2.,3.]);s,m,d=predict_measurements(state,intervals,340.)
    np.testing.assert_allclose(s[0,0],(np.sqrt(5)-2)/340+2*(1+1e-5),atol=1e-14)
    np.testing.assert_allclose(m[0,0],1/340+.002,atol=1e-14)
    np.testing.assert_allclose(m[0,1],(2-np.sqrt(5))/340+.002+2*(-3e-5),atol=1e-14)
    np.testing.assert_array_equal(d,[[1,0,0],[-1,1,0]])

def test_analytic_jacobian_against_independent_finite_differences():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:state=f['final']
    p=pack(state);r,j=residual_jacobian(state,data);fd=np.empty_like(j)
    for k in range(len(p)):
        h=1e-6*max(1,abs(p[k]));a=p.copy();b=p.copy();a[k]+=h;b[k]-=h
        fd[:,k]=(residual_jacobian(unpack(a),data)[0]-residual_jacobian(unpack(b),data)[0])/(2*h)
    np.testing.assert_allclose(j,fd,rtol=1e-6,atol=5e-9)

def test_normal_step_solves_weighted_linearized_equations():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:j=f['J_final'];r=f['residual_final']
    delta=normal_step(j,r);w=np.r_[np.full(256,1e8),np.full(39,1e4)]
    # Scale by each normal-equation diagonal to assess all units fairly.
    normal=j.T@(w[:,None]*j);rhs=j.T@(w*r)
    assert np.linalg.norm((normal@delta+rhs)/np.sqrt(np.diag(normal)))<1e-7
