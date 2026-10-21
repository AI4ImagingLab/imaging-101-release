import numpy as np
from src.physics_model import grid,flux_divergence
from src.solvers import solve
from src.preprocessing import balance_flux
def test_analytic_dipole_and_flux_laws():
    ns,np_,nr=60,120,35;g=grid(ns,np_,nr,2.5)
    boundary=np.broadcast_to(g['sc'][:,None],(ns,np_)).copy();out=solve(boundary,nr,2.5)
    radii=np.exp(g['rg']);expected=g['sc'][None,:,None]*(1+2*(2.5/radii[:,None,None])**3)/(1+2*2.5**3)
    relative=np.linalg.norm(out['radial']-expected)/np.linalg.norm(np.broadcast_to(expected,out['radial'].shape))
    assert relative<.03
    assert np.max(abs(flux_divergence(out,g)))<1e-12
    assert np.max(abs(out['azimuthal']))<1e-12
    np.testing.assert_allclose(out['radial'][0],boundary,rtol=1e-10,atol=1e-12)
    for shell in out['radial']:assert abs(shell.sum())<1e-9
def test_linearity_and_periodic_seam():
    ns,nphi,nr=20,40,8;g=grid(ns,nphi,nr,2.5);phi=(np.arange(nphi)+.5)*g['dp']
    boundary=balance_flux(g['sc'][:,None]*np.cos(phi)[None])
    a=solve(boundary,nr);b=solve(-3*boundary,nr)
    assert np.max(abs(flux_divergence(a,g)))<1e-12
    for k in a:np.testing.assert_allclose(b[k],-3*a[k],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(a['azimuthal'][:,:,0],a['azimuthal'][:,:,-1],atol=1e-14)
def test_metrics_known_cases():
    from src.visualization import metrics
    assert metrics([0,1],[0,1])==dict(ncc=1.,nrmse=0.)
    assert metrics([1,0],[0,1])==dict(ncc=0.,nrmse=1.)
