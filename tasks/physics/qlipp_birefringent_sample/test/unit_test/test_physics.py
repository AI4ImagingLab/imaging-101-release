import numpy as np,torch
from src.physics_model import stokes_forward,estimate_parameters,apply_matrix,mueller_matrix,inverse_mueller

def test_independent_complex_jones_retarder():
    a=.37;delta=.42;c,s=np.cos(a),np.sin(a);R=np.array([[c,-s],[s,c]])
    J=R@np.diag(np.exp(1j*np.array([delta,-delta])/2))@R.T
    Ex,Ey=J@np.array([1,-1j])/np.sqrt(2)
    actual=np.array([abs(Ex)**2+abs(Ey)**2,abs(Ex)**2-abs(Ey)**2,2*(Ex*Ey.conjugate()).real,2*(Ex*Ey.conjugate()).imag])
    expected=stokes_forward(torch.tensor(delta),torch.tensor(a),torch.tensor(1.),torch.tensor(1.)).numpy()
    np.testing.assert_allclose(expected,actual,rtol=2e-7,atol=2e-7)

def test_roundtrip_and_depolarization_conservation():
    d=torch.tensor([.1,.2,.5]);a=torch.tensor([.2,.7,2.]);t=torch.tensor([.4,.8,1.]);p=torch.tensor([.7,.9,1.]);s=stokes_forward(d,a,t,p)
    for est,true in zip(estimate_parameters(s),[d,a,t,p]):torch.testing.assert_close(est,true,rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(torch.linalg.vector_norm(s[1:],dim=0),t*p)

def test_background_inverse_identity():
    s=stokes_forward(torch.tensor([.2]),torch.tensor([.6]),torch.tensor([.8]),torch.tensor([1.]))
    m=mueller_matrix(s);inv=inverse_mueller(s)
    np.testing.assert_allclose(torch.einsum('ijb,jkb->ikb',m,inv).numpy()[:,:,0],np.eye(4),atol=2e-7)

def test_pi_periodic_slow_axis():
    p=[torch.tensor([.2]),torch.tensor([.4]),torch.tensor([.8]),torch.tensor([.9])]
    torch.testing.assert_close(stokes_forward(*p),stokes_forward(p[0],p[1]+np.pi,p[2],p[3]),atol=1e-7,rtol=1e-6)

def test_visualization_circular_metric_known_case():
    from src.visualization import angular_error
    assert angular_error(np.array([.2]),np.array([.2+np.pi]),np.array([.1]))<1e-14
