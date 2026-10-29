from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_observations, observation_vector
from src.physics_model import (build_green_matrix, build_data_covariance,
                               build_model_covariance, mogi_deflation_basis,
                               displacement)

ROOT=Path(__file__).resolve().parents[2]


def test_full_bound_preprocessing_and_covariances():
    data=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as source:
        assert_allclose(observation_vector(data),source['_observations'],atol=0,rtol=0)
        assert_allclose(build_data_covariance(data),source['_data_covariance'],atol=1e-11,rtol=2e-12)
    with np.load(ROOT/'test/fixtures/original_system.npz',allow_pickle=False) as source:
        assert_allclose(build_model_covariance(data['fault_triangles']),source['model_covariance'],atol=1e-11,rtol=2e-12)


def test_all_actual_green_entries_against_original_formulas():
    data=load_observations(ROOT/'data/raw_data.npz')
    actual=build_green_matrix(data)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as source:
        assert_allclose(actual,source['_green'],atol=2e-11,rtol=1e-9)


def test_independent_mogi_units_radial_direction_and_symmetry():
    #1kPa pressure has1000Pa; formula returns metres, then1000mm/m.
    from src.physics_model import BASIN_CENTER_KM, BASIN_DEPTH_M, BASIN_RADIUS_M, SHEAR_MODULUS_PA
    xy=np.array([BASIN_CENTER_KM,np.array(BASIN_CENTER_KM)+[1.,0.],np.array(BASIN_CENTER_KM)-[1.,0.]])
    basis=mogi_deflation_basis(xy)
    expected=-1000.*BASIN_RADIUS_M**3*1000.*.75/(SHEAR_MODULUS_PA*BASIN_DEPTH_M**2)
    assert_allclose(basis[0],[0.,0.,expected],rtol=2e-15,atol=1e-14)
    assert_allclose(basis[1,0],-basis[2,0],rtol=2e-14)
    assert_allclose(basis[1,2],basis[2,2],rtol=2e-14)
    assert basis[1,0]<0 and basis[1,2]<0


def test_independent_elastic_linearity_vertex_orientation_and_translation():
    triangle=np.array([[0.,0.,2.],[4.,1.,3.],[1.,3.,6.]])
    x=np.array([-2.,2.,7.]);y=np.array([3.,-4.,1.]);zero=np.zeros(3)
    calc=lambda p,s:np.column_stack(displacement(x,y,zero,list(p),s,0.,0.))
    unit=calc(triangle,1.)
    assert_allclose(calc(triangle,2.),2.*unit,rtol=1e-12,atol=2e-14)
    assert_allclose(calc(triangle[[0,2,1]],1.),unit,rtol=1e-12,atol=2e-14)
    translated=triangle+[10.,-8.,0.]
    shifted=np.column_stack(displacement(x+10.,y-8.,zero,list(translated),1.,0.,0.))
    assert_allclose(shifted,unit,rtol=1e-12,atol=2e-14)
