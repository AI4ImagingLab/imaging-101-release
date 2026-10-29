from pathlib import Path
import importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('entropy',ROOT/'evaluation/entropy_contract.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
def test_dipole_normalization_and_ring_inversion():
 theta=[];phi=[]
 for i in range(60):
  t=np.pi*(i+.5)/60;n=round(120*np.sin(t));theta.extend([t]*n);phi.extend(2*np.pi*(np.arange(n)+.5)/n)
 theta=np.asarray(theta);phi=np.asarray(phi);c=np.zeros((3,135),complex);c[0,0]=1
 f=e.field(c,theta,phi)
 np.testing.assert_allclose(f[0],np.sqrt(3/(4*np.pi))*np.cos(theta),atol=2e-15)
 np.testing.assert_allclose(f[1:],0,atol=1e-15)
 np.testing.assert_allclose(e.coefficients(f,theta,phi),c,atol=1e-13)
 expected=np.arcsinh(1/200)-np.sqrt(40001)+200
 assert abs(e.entropy_cost(c)-expected)<1e-12
