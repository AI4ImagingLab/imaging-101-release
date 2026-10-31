import numpy as np
from src.physics_model import *
from src.preprocessing import prepare

def test_steady_sliding_force_and_state():
    a=.01;b=.0099;D=3e-6;mu0=.693
    initial,offset=initial_conditions(a,b,D,mu0)
    np.testing.assert_allclose(initial,[0,INITIAL_VELOCITY,np.log(10)],atol=1e-13,rtol=0)
    derivative=derivatives(0,initial,a,b,D,mu0,offset)
    assert abs(derivative[1])<1e-9 and abs(derivative[2])<1e-14
    assert state_rhs(0,initial[2],D)==0
    assert state_rhs(REFERENCE_VELOCITY,0,D)==0

def test_loading_integral_and_constitutive_inverse():
    assert loadpoint(0)==0
    for time in [0.,10.,24.8,50.]:
        numerical=(loadpoint(time+1e-4)-loadpoint(time-1e-4))/2e-4
        np.testing.assert_allclose(numerical,loadpoint_velocity(time),atol=2e-14,rtol=1e-7)
        assert loadpoint_velocity(time)>0
    a=.01;b=.0099;mu0=.693;v=3e-7;psi=2.3
    mu=regularized_friction(v,psi,a,b,mu0)
    recovered=(a*np.log(np.sinh(mu/a)*2*REFERENCE_VELOCITY/v)-mu0)/b
    assert abs(recovered-psi)<1e-13

def test_machine_units_and_dimensional_values():
    assert STIFFNESS==90.5e6/.001
    assert NORMAL_STRESS==(60-20)*1e6
    assert MASS_PER_AREA==7798.2
    assert abs(DAMPING-25.52645686341918)<1e-12

def test_exact_state_jacobian_independent_centered_difference():
    state=np.array([1e-5,2e-6,.7]);a=.01;b=.0099;D=3e-6;mu0=.693;offset=.0003
    actual=state_jacobian(20,state,a,b,D,mu0,offset)
    columns=[]
    for i,h in enumerate([1e-8,1e-10,1e-5]):
        delta=np.zeros(3);delta[i]=h
        columns.append((derivatives(20,state+delta,a,b,D,mu0,offset)-derivatives(20,state-delta,a,b,D,mu0,offset))/(2*h))
    np.testing.assert_allclose(actual,np.array(columns).T,rtol=1e-6,atol=1e-7)
