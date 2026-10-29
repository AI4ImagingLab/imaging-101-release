import numpy as np
from src.physics_model import residual_jacobians,weighted_error,wrap
def toy():
    return {'edge_indices':np.array([[0,1]]),'relative_poses':np.array([[1.,0.,np.pi/2]]),'information':np.eye(3)[None]}
def test_known_group_composition():
    d=toy(); x=np.array([[0.,0.,np.pi/2],[0.,1.,-np.pi]])
    e,_,_=residual_jacobians(x,d);np.testing.assert_allclose(e,0,atol=1e-15)
def test_analytic_jacobians_and_gradient():
    d=toy(); x=np.array([[.2,.5,.4],[1.2,-.3,-.7]])
    e,a,b=residual_jacobians(x,d)
    for node,j in [(0,a),(1,b)]:
        for k in range(3):
            xp=x.copy();xm=x.copy();xp[node,k]+=1e-6;xm[node,k]-=1e-6
            fd=(residual_jacobians(xp,d)[0]-residual_jacobians(xm,d)[0])/2e-6
            np.testing.assert_allclose(fd[0],j[0,:,k],atol=3e-10)
            fd_cost=(weighted_error(xp,d)-weighted_error(xm,d))/2e-6
            np.testing.assert_allclose(fd_cost,2*j[0,:,k]@e[0],atol=2e-9)
def test_gauge_invariance():
    d=toy(); x=np.array([[.2,.5,.4],[1.2,-.3,-.7]]); t=.9
    y=x.copy();y[:,:2]=x[:,:2]@np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])+[3,4];y[:,2]=wrap(x[:,2]+t)
    np.testing.assert_allclose(weighted_error(x,d),weighted_error(y,d),atol=1e-13)
def test_known_weighted_metric():
    d=toy();d['relative_poses'][:]=0;d['information'][0]=np.diag([2,3,4]);x=np.array([[0.,0.,0.],[1.,2.,.5]])
    assert weighted_error(x,d)==15

def test_notebook_metrics_known_case():
    from src.visualization import comparison_metrics
    d=toy();d['relative_poses'][:]=0;d['information'][0]=np.diag([2,3,4]);x=np.array([[0.,0.,0.],[1.,2.,.5]]);ref=x.copy();ref[1]=[.4,1.2,.25]
    assert comparison_metrics(x,ref,d)=={'position_max_error_m':1.,'heading_max_error_rad':.25,'weighted_observation_error':15.}
