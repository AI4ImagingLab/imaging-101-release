from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def test_preprocessing_whole_original_observations():
 from src.preprocessing import prepare
 with np.load(ROOT/'data/raw_data.npz') as raw,np.load(ROOT/'test/fixtures/original_stages.npz') as original:
  data=prepare(raw)
  np.testing.assert_array_equal(data,original['preprocessed'])

def test_complete_wet_dry_reference_and_power_stages():
 from src.preprocessing import prepare
 from src.physics_model import retrieve_links
 with np.load(ROOT/'data/raw_data.npz') as raw,np.load(ROOT/'test/fixtures/original_stages.npz') as original:
  d=prepare(raw);result=retrieve_links(d,raw['projected_links'][0],raw['rain_coefficients'][0])
  for key in ['Dry','F','Pref','PminCor','PmaxCor','Rmean']:
   np.testing.assert_allclose(result[key],original[key],rtol=0,atol=1e-7,equal_nan=True,err_msg=key)

def test_original_midpoints_variogram_and_full_kriging_grids():
 from src.solvers import midpoint_observations,interpolate,climatological_variogram
 with np.load(ROOT/'data/raw_data.npz') as raw,np.load(ROOT/'test/fixtures/original_stages.npz') as original,np.load(ROOT/'test/fixtures/original_kriging.npz') as case:
  d=original['preprocessed']
  for stamp in [201109100815,201109101200,201109110000]:
   points,values=midpoint_observations(d,original['Rmean'],raw['projected_links'][0],stamp)
   np.testing.assert_allclose(points,case[f'points_{stamp}'],rtol=0,atol=1e-9)
   np.testing.assert_allclose(values,case[f'rain_{stamp}'],rtol=0,atol=1e-10)
   field=interpolate(points,values,raw['grid_xy'][0],stamp)
   np.testing.assert_allclose(field,case[f'field_{stamp}'],rtol=0,atol=1e-7)
  sill,nugget,radius=climatological_variogram(201109100815)
  np.testing.assert_allclose([sill,nugget,radius],[2.2801847003399,.22801847003399,25.4533444221877],atol=1e-12,rtol=0)

def test_complete_raw_to_whole_daily_field():
 from src.solvers import reconstruct
 with np.load(ROOT/'data/raw_data.npz') as raw,np.load(ROOT/'test/fixtures/original_stages.npz') as original:
  fresh=reconstruct(raw);x=fresh['reconstruction'];y=original['reconstruction']
  error=float(np.sqrt(np.mean((x-y)**2))/np.ptp(y))
  assert error<=9.663465651076436e-9
  assert x.shape==(1,38063) and np.isfinite(x).all() and np.min(x)>=0
  assert len(fresh['timestamps'])==96 and fresh['timestamps'][0]==201109100815 and fresh['timestamps'][-1]==201109110800
  unsupported=~np.isfinite(fresh['stages']['Dry'])
  assert not np.isfinite(fresh['stages']['Rmean'][unsupported]).any()
