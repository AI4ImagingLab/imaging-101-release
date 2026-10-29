import numpy as np
from numpy.testing import assert_allclose
from src.solvers import objective_gradient,tie_deep_parameters,output_arrays
from src.generate_data import correlated_velocity_perturbation


def test_independent_quadratic_value_gradient_finite_difference():
    g=np.array([[1.,2.],[-1.,3.],[2.,-.5]])
    d=np.array([2.,-1.,.4]);a=np.diag([1.,2.,3.]);b=np.array([[2.,.3],[.3,1.]])
    prior=np.array([-.2,.5]);x=np.array([.3,-.4])
    f,gradient=objective_gradient(x,g,d,a,b,prior)
    assert_allclose(f,.5*(g@x-d)@a@(g@x-d)+.5*(x-prior)@b@(x-prior))
    for i in range(2):
        step=np.eye(2)[i]*1e-6
        numerical=(objective_gradient(x+step,g,d,a,b,prior)[0]-objective_gradient(x-step,g,d,a,b,prior)[0])/2e-6
        assert_allclose(numerical,gradient[i],rtol=2e-9,atol=2e-9)


def test_exact_deep_equality_and_source_variance_convention():
    g=np.arange(2*1305,dtype=float).reshape(2,1305);cm=np.eye(1305)*7.
    reduced,c,mapping=tie_deep_parameters(g,cm)
    assert reduced.shape==(2,1301) and c.shape==(1301,1301)
    assert_allclose(reduced,g@mapping,atol=0,rtol=0)
    assert c[-1,-1]==100. and c[0,0]==7.
    solution=mapping@np.arange(1301,dtype=float)
    assert_allclose(solution[1292:1297],1300.,atol=0,rtol=0)


def test_all_output_quantities_and_units_partition():
    m=np.arange(1305,dtype=float);a=output_arrays(m)
    assert a['slip_rate'].shape==(1,1299)
    assert a['basin_coefficient'][0,0]==1299.
    assert_allclose(a['datum_offset'][0],np.arange(1300,1305))


def test_diagnostic_generator_seed_units_axis_order_and_input_immutable():
    data={'sar_velocity':np.zeros(3476),'gnss_velocity':np.zeros((39,2))}
    covariance=np.eye(3554)*4.
    a,noise=correlated_velocity_perturbation(data,covariance,seed=11,fraction=.25)
    expected=.5*np.random.default_rng(11).standard_normal(3554)
    assert_allclose(noise,expected,atol=0,rtol=0)
    assert_allclose(a['sar_velocity'],expected[:3476],atol=0,rtol=0)
    assert_allclose(a['gnss_velocity'][:,1],expected[3515:],atol=0,rtol=0)
    assert np.all(data['sar_velocity']==0) and np.all(data['gnss_velocity']==0)
