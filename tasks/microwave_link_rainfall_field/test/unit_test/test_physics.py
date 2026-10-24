import numpy as np
import pytest
from src.solvers import spherical_variogram,interpolate,climatological_variogram

def test_spherical_variogram_units_limits_and_nugget():
 d=np.array([0,1e-12,5,10,20]);v=spherical_variogram(d,4,1,10)
 np.testing.assert_allclose(v,[0,1+6e-13,3.75,5,5],atol=1e-14,rtol=0)
 assert np.all(np.diff(v)>=0)

def test_constant_field_reproduction_and_nonnegative_projection():
 p=np.array([[0.,0.],[0,5],[5,0],[5,5]]);grid=np.array([[0,0],[2.5,2.5],[100,100]])
 for method in ['OK','IDW']:
  np.testing.assert_allclose(interpolate(p,np.full(4,3.),grid,201109102045,method=method),3,atol=1e-13,rtol=0)
 assert np.all(interpolate(p,np.full(4,-3.),grid,201109102045)==0)

def test_midpoint_duplicate_direction_averaging():
 from src.solvers import midpoint_observations
 data=np.zeros((3,11));data[:,1]=201109102045;data[:,9]=[1,2,3]
 geometry=np.array([[1,0,0,2,0],[2,2,0,0,0],[3,4,0,6,0]])
 p,r=midpoint_observations(data,np.array([2.,4.,8.]),geometry,201109102045)
 np.testing.assert_array_equal(p,[[1,0],[5,0]]);np.testing.assert_array_equal(r,[3,8])

def test_bad_method_and_absent_observations():
 from src.solvers import midpoint_observations
 with pytest.raises(ValueError):interpolate(np.zeros((1,2)),np.ones(1),np.zeros((1,2)),201109102045,method='invalid')
 with pytest.raises(ValueError):midpoint_observations(np.zeros((1,11)),np.array([np.nan]),np.zeros((1,5)),201109102045)

def test_climatological_range_conversion_and_day_boundary():
 a=climatological_variogram(201109102345);b=climatological_variogram(201109110000)
 assert a[2]==pytest.approx(25.4533444221877) and b[2]==pytest.approx(25.7598326971477)
 assert a[1]==pytest.approx(.1*a[0]) and a!=b

def test_power_to_rain_independent_dimensional_known_case():
 from src.physics_model import rainfall_from_powers
 d=np.zeros((3,11));d[:,0]=38;d[:,4]=2
 table=np.array([[10.,2,1],[100.,2,1]]);coeff=np.stack([table,table])
 r=rainfall_from_powers(d,np.full(3,-50.),np.array([-56.3,-52.3,-50]),np.array([-52.3,-50,-50]),coeff)
 np.testing.assert_allclose(r,[1.32,0,0],atol=1e-14,rtol=0)

def test_reference_timestamp_and_public_acquisition_contract():
 from pathlib import Path
 root=Path(__file__).resolve().parents[2]
 with np.load(root/'notebooks/reference_outputs/original_instantaneous.npz') as z:
  assert int(z['timestamp'])==201109102045 and z['rainfall'].shape==(38063,)
 with np.load(root/'data/raw_data.npz') as z:
  expected=dict(link_records=((1,496180,11),'float64'),record_valid=((1,496180,11),'uint8'),projected_links=((1,2472,5),'float64'),grid_xy=((1,38063,2),'float64'),rain_coefficients=((1,2,201,3),'float64'))
  assert set(z.files)==set(expected)
  for key,(shape,dtype) in expected.items():assert z[key].shape==shape and z[key].dtype==np.dtype(dtype) and np.isfinite(z[key]).all()
  assert np.all((z['record_valid']==0)|(z['record_valid']==1))
 text=(root/'README.md').read_text(encoding='utf8')
 for key in expected:assert '| '+key+' |' in text
 for value in ['0.25','10','4326','28992','12.5','40.5','2011-09-10','2011-09-11']:assert value in text
