"""Explicit stress-based spring-slider and regularized slip-state friction law."""
import numpy as np
from .ode_integrator import integrate
from scipy.interpolate import interp1d
MASS_PER_AREA=7798.2
STIFFNESS=90.5e9
NORMAL_STRESS=40e6
REFERENCE_VELOCITY=3e-6
INITIAL_VELOCITY=3e-7
JUMP_TIME=24.8
DAMPING=3e-6*2*np.sqrt(STIFFNESS*200)

def logcosh(value):
    return abs(value)+np.log1p(np.exp(-2*abs(value)))-np.log(2)

def loadpoint(time):
    return INITIAL_VELOCITY*time+2.7e-6*logcosh(time-JUMP_TIME)/2+2.7e-6*time/2-2.7e-6*logcosh(-JUMP_TIME)/2

def loadpoint_velocity(time):
    return INITIAL_VELOCITY+2.7e-6*np.tanh(time-JUMP_TIME)/2+2.7e-6/2

def regularized_friction(velocity,psi,a,b,mu0):
    return a*np.arcsinh(velocity/(2*REFERENCE_VELOCITY)*np.exp((mu0+b*psi)/a))

def state_rhs(velocity,psi,Dc):
    # Continuous zero-velocity limit completes V*log(V) without NaN.
    return 0. if velocity<=0 else -velocity/Dc*(np.log(velocity/REFERENCE_VELOCITY)+psi)

def initial_conditions(a,b,Dc,mu0,mass=MASS_PER_AREA):
    mu=regularized_friction(INITIAL_VELOCITY,np.log(10),a,b,mu0)
    psi=(a*np.log(np.sinh(mu/a)*2*REFERENCE_VELOCITY/INITIAL_VELOCITY)-mu0)/b
    loading_offset=mu*NORMAL_STRESS/STIFFNESS+DAMPING/STIFFNESS*INITIAL_VELOCITY
    return np.array([0.,INITIAL_VELOCITY,psi]),loading_offset

def derivatives(time,state,a,b,Dc,mu0,loading_offset,mass=MASS_PER_AREA):
    displacement,velocity,psi=state
    friction=regularized_friction(velocity,psi,a,b,mu0)
    return np.array([velocity,STIFFNESS/mass*(loadpoint(time)+loading_offset-displacement)-NORMAL_STRESS*friction/mass-DAMPING*velocity/mass,state_rhs(velocity,psi,Dc)])

def state_jacobian(time,state,a,b,Dc,mu0,loading_offset,mass=MASS_PER_AREA):
    """Exact active-state derivative, matching source stiff-solver AD role."""
    _,v,psi=state
    z=v/(2*REFERENCE_VELOCITY)*np.exp((mu0+b*psi)/a)
    ratio=z/np.hypot(1,z)
    mu_v=a*ratio/v
    mu_psi=b*ratio
    if v>0:state_v=-(np.log(v/REFERENCE_VELOCITY)+psi+1)/Dc;state_psi=-v/Dc
    else:state_v=state_psi=0.
    return np.array([[0.,1.,0.],[-STIFFNESS/mass,-NORMAL_STRESS*mu_v/mass-DAMPING/mass,-NORMAL_STRESS*mu_psi/mass],[0.,state_v,state_psi]])

def trajectory(log_parameters,data,mass=MASS_PER_AREA):
    a,Dc=10**np.asarray(log_parameters);b=a+data['b_minus_a'];mu0=data['mu0']
    initial,offset=initial_conditions(a,b,Dc,mu0,mass)
    times,states,statistics=integrate(lambda t,u:derivatives(t,u,a,b,Dc,mu0,offset,mass),lambda t,u:state_jacobian(t,u,a,b,Dc,mu0,offset,mass),lambda t,u:np.array([0.,STIFFNESS/mass*loadpoint_velocity(t),0.]),initial,data['time'][-1])
    if not np.all(states[1]>0):raise ValueError('Nonpositive slider velocity outside selected stable trajectory')
    v=states[1];psi=states[2];mu=mu0+a*np.log(v/REFERENCE_VELOCITY)+b*psi
    coordinate=loadpoint(times)*1000
    friction=interp1d(coordinate,mu,fill_value='extrapolate')(data['displacement'])
    velocity=interp1d(coordinate,v,fill_value='extrapolate')(data['displacement'])
    state=interp1d(coordinate,psi,fill_value='extrapolate')(data['displacement'])
    return {'reconstruction':np.array([[a,b,Dc]]),'friction':friction[None],'slider_velocity':velocity[None],'state':(Dc/REFERENCE_VELOCITY*np.exp(state))[None]}
