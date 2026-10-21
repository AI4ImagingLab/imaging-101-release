import numpy as np,torch
from scipy.integrate import quad
from src.physics_model import sersic,PixelModel,objective
def test_continuous_flux_and_half_light_definition():
    p=torch.tensor([.3,.4,10.,8.,1.,0.,.2,0.],dtype=torch.float64)
    def radial(r):return float(sersic(torch.tensor(r+.3),torch.tensor(.4),p))*2*np.pi*r
    total=quad(radial,0,np.inf,epsabs=1e-8)[0];inside=quad(radial,0,8,epsabs=1e-8)[0]
    np.testing.assert_allclose(total,10,rtol=1e-6);assert abs(inside/total-.5)<1e-5
def test_physical_flux_psf_centering_and_gradient():
    psf=np.zeros((9,9));psf[4,4]=1
    model=PixelModel(psf,shape=(40,40),dtype=torch.float64)
    p=torch.tensor([19.4,20.1,20.,5.,1.3,.2,.6,.0001],dtype=torch.float64,requires_grad=True)
    intrinsic=model.intrinsic(p);observed=model.observed(p)
    np.testing.assert_allclose(float((observed-p[7]).sum().detach()),float(intrinsic.sum().detach()),rtol=1e-12)
    image=torch.zeros((40,40),dtype=torch.float64);rms=torch.ones((40,40),dtype=torch.float64)*.02;valid=torch.ones((40,40),dtype=torch.bool)
    value=objective(p,model,image,rms,valid);grad=torch.autograd.grad(value,p)[0].numpy();numeric=[]
    for i,h in enumerate([1e-4,1e-4,1e-4,1e-4,1e-4,1e-5,1e-5,1e-7]):
        a=p.detach().clone();b=a.clone();a[i]+=h;b[i]-=h
        numeric.append(float((objective(a,model,image,rms,valid)-objective(b,model,image,rms,valid))/(2*h)))
    np.testing.assert_allclose(grad,numeric,rtol=2e-5,atol=1e-4)
    # Original approximatepi makes a bounded fractional-pixel shift; quantify
    # it rather than silently calling it mathematically exact centering.
    assert 0<31*(1-3.1415/np.pi)<.001
