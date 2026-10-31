import numpy as np
from src.physics_model import free_energy,unfolded_fraction,baseline_planes,fluorescence,R
from src.preprocessing import normalize_intensity
def test_midpoint_and_denaturant_sign():
 p=np.array([50.,350.,5.,14.]);assert free_energy(50.,0.,p)==0.;assert unfolded_fraction(50.,0.,p)==.5
 np.testing.assert_allclose(free_energy(25.,2.,p)-free_energy(25.,1.,p),-14.);assert unfolded_fraction(25.,2.,p)>unfolded_fraction(25.,1.,p)
def test_gibbs_helmholtz_and_heat_capacity():
 p=np.array([50.,350.,5.,14.]);h=.01;T=50.+273.15;g0=free_energy(50.,0.,p);plus=free_energy(50.+h,0.,p);minus=free_energy(50.-h,0.,p);derivative=(plus-minus)/(2*h);curvature=(plus-2*g0+minus)/h**2
 np.testing.assert_allclose(g0-T*derivative,350.,rtol=1e-8);np.testing.assert_allclose(curvature,-5./T,rtol=2e-7)
def test_baseline_and_population_formula():
 p=np.zeros(18);p[0]=2.;p[3]=6.;p[7]=3.;p[10]=9.;p[14:]=[50.,350.,5.,14.]
 np.testing.assert_allclose(fluorescence(p,np.array([50.,50.]),np.zeros(2),np.array([0,1])),[4.,6.])
 F,U=baseline_planes(np.array([26.]),np.array([2.]),np.arange(7.));np.testing.assert_allclose(F,[4.]);np.testing.assert_allclose(U,[22.])
def test_observed_scale_gauge():
 t=np.array([[69.,71.,72.]]);y=np.array([[[2.,4.,8.]],[[3.,6.,12.]]]);a=normalize_intensity(t,y);np.testing.assert_allclose(a,[[[1.,2.,4.]],[[1.,2.,4.]]]);np.testing.assert_allclose(normalize_intensity(t,7*y),a)
