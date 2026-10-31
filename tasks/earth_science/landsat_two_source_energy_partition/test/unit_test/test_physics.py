import numpy as np
from src import physics_model as p
def test_independent_atmospheric_units_and_limits():
    t=np.array([300.]);pressure=np.array([1000.]);vapor=np.array([0.])
    np.testing.assert_allclose(p.met_calc_rho(pressure,vapor,t),100000/(287.04*300))
    np.testing.assert_allclose(p.met_calc_c_p(pressure,vapor),1003.5)
    np.testing.assert_allclose(p.met_calc_stephan_boltzmann(t),5.670373e-8*300**4)
    np.testing.assert_allclose(p.MO_calc_Psi_H(np.array([0.])),0)
    np.testing.assert_allclose(p.MO_calc_Psi_M(np.array([0.])),0)
    np.testing.assert_allclose(p.MO_calc_u_star(np.array([3.]),10.,np.array([np.inf]),.195,.0375),3*.41/np.log((10-.195)/.0375))
    assert p.MO_calc_L(np.array([.3]),t,np.array([1.2]),np.array([1004.]),np.array([100.]),np.array([0.]))[0]<0
    assert p.MO_calc_L(np.array([.3]),t,np.array([1.2]),np.array([1004.]),np.array([-100.]),np.array([0.]))[0]>0
    assert np.isinf(p.MO_calc_L(np.array([.3]),t,np.array([1.2]),np.array([1004.]),np.array([0.]),np.array([0.]))[0])
def test_independent_fourth_power_temperature_inversion():
    f=np.array([.1,.4,.8]);tc=np.array([295.,298.,301.]);ts=np.array([310.,312.,315.]);tr=(f*tc**4+(1-f)*ts**4)**.25
    flag,recovered=p.TSEB_calc_T_S(tr,tc,f)
    np.testing.assert_array_equal(flag,0);np.testing.assert_allclose(recovered,ts,atol=3e-5,rtol=0)
    badflag,_=p.TSEB_calc_T_S(np.array([100.]),np.array([400.]),np.array([.9]))
    assert badflag[0]==255
def test_corrected_weiss_equations_horizontal_units():
    angles=np.array([0.,30.,60.,80.]);pressure=np.full(4,1013.25);mu=np.cos(np.deg2rad(angles));m=1/mu
    vis=1320*(1-.5455);nir=1320*.5455
    water=1320*10**(-1.195+.4459*np.log10(m)-.0345*np.log10(m)**2)
    beamvis=vis*np.exp(-.185*m)*mu;beamnir=np.maximum(0,(nir*np.exp(-.06*m)-water)*mu)
    expected=[beamvis,.4*(vis*mu-beamvis),beamnir,np.maximum(0,.6*((nir-water)*mu-beamnir))]
    for got,ref in zip(p.rad_calc_potential_irradiance_weiss(angles,pressure),expected):np.testing.assert_allclose(got,ref,rtol=1e-13,atol=1e-12)
    assert np.all(np.diff(water)>0) # Increasing optical path increases absorption.
    assert np.all(sum(expected)<=1320*mu) and all(np.all(x>=0) for x in expected)
    # Sea-level optical attenuation is exp(-coefficient*m), not pressure/1313.25.
    np.testing.assert_allclose(expected[0][0]/vis,np.exp(-.185))
def test_resistance_network_and_crop_limit():
    lai=np.array([.1,1.,3.]);h=np.full(3,.3)
    z,d=p.res_calc_roughness(lai,h);np.testing.assert_allclose(z,.0375);np.testing.assert_allclose(d,.195)
    np.testing.assert_allclose(p.res_calc_R_x_Norman(lai,np.full(3,.1),np.ones(3)),90/lai*np.sqrt(.1))
    np.testing.assert_allclose(p.res_calc_R_S_Kustas(np.ones(3),np.zeros(3)),1/.012)
