from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare
from src.physics_model import covariance
from src.solvers import likelihood,reconstruct,predict
ROOT=Path(__file__).resolve().parents[2]
def test_frozen_stages():
 f=np.load(ROOT/'test/fixtures/upstream.npz');d=prepare(load_data(ROOT/'data/raw_data.npz'))
 np.testing.assert_array_equal(d['time'],f['sorted_times']);np.testing.assert_array_equal(d['channel'],f['sorted_channels']);np.testing.assert_array_equal(d['error'],f['sorted_errors'])
 for label,key in [('initial','initial'),('','parameters')]:
  prefix=label+'_' if label else '';p=f[key]
  np.testing.assert_allclose(covariance(d['time'],d['channel'],d['time'],d['channel'],p),f[prefix+'covariance'],rtol=1e-9,atol=1e-8)
  components,q=likelihood(p,d,components=True)
  np.testing.assert_allclose(components,f[prefix+'components'],rtol=1e-9,atol=1e-8);np.testing.assert_allclose(q,f[prefix+'mean_correction'],rtol=1e-9,atol=1e-8)
def test_native_prediction():
 f=np.load(ROOT/'test/fixtures/upstream.npz');d=prepare(load_data(ROOT/'data/raw_data.npz'));m,sd=predict(f['parameters'],d,f['time'])
 np.testing.assert_allclose(m,f['reconstruction'],rtol=1e-9,atol=1e-8);np.testing.assert_allclose(sd,f['conditional_sd'],rtol=1e-9,atol=1e-8)
def test_full_pipeline():
 f=np.load(ROOT/'test/fixtures/upstream.npz');out,info=reconstruct(load_data(ROOT/'data/raw_data.npz'))
 assert np.max(abs((info['parameters']-f['parameters'])/np.array([1,1,10,10,1,10,10,1])))<.005
 assert np.max(np.sqrt(np.mean((out['reconstruction'][0]-f['reconstruction'])**2,axis=1)))<.01
