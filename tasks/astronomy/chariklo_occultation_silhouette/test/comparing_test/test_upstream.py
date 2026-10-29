from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_data,radial_uncertainty
from src.physics_model import ellipse_contour,contact_chi2,negative_margin,contact_time_residuals
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_original_objectives_contour_and_complete_pipeline():
    data=load_data()
    with np.load(ROOT/'test/fixtures/upstream_cases.npz') as z:
        assert_allclose(contact_chi2(z['trial_parameters'],data['limb_points'],radial_uncertainty(data)),z['trial_chi2'],rtol=1e-11)
        assert_allclose(ellipse_contour(z['parameters'],z['theta']),z['contour'],atol=1e-11)
        out,stages=reconstruct(data)
        assert_allclose(out['parameters'][0],z['parameters'],atol=1e-10)
        assert_allclose(out['reconstruction'][0],z['contour'],atol=1e-11)
    assert len(data['limb_points'])==10
    assert len(stages['chi2'])==20621
    assert stages['negative_clear'].sum()==19776

def test_known_circle_contacts_and_negative_chord():
    p=np.array([0.,0.,2.,0.,73.]);theta=np.linspace(0,2*np.pi,20)
    contour=ellipse_contour(p,theta)
    assert_allclose(np.linalg.norm(contour,axis=1),2,atol=1e-15)
    assert_allclose(contact_chi2(p,contour,np.ones(20)),0,atol=1e-28)
    assert_allclose(negative_margin(p,np.array([[-3.,3.],[0.,3.],[3.,3.]])),1.)
    assert negative_margin(p,np.array([[0.,1.]]))<0

def test_quadratic_contact_prediction_independent_case():
    p=np.array([0.,0.,2.,0.,0.])
    points=np.array([[-2.2,0.],[1.7,0.]])
    velocity=np.array([[1.,0.],[1.,0.]])
    assert_allclose(contact_time_residuals(p,points,velocity),[.2,.3],atol=1e-14)
    # A rotated ellipse obeys the corresponding quadratic form exactly.
    p=np.array([1.,-2.,3.,.2,37.]);theta=np.linspace(0,2*np.pi,10)
    curve=ellipse_contour(p,theta);v=np.tile([1.,.2],(10,1))
    assert_allclose(contact_time_residuals(p,curve,v),0,atol=3e-14)
