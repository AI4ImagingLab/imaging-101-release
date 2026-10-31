from pathlib import Path
import sys
import numpy as np
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T))
from src.preprocessing import load
from src.physics_model import gaussian_model,K,REF_FREQ,NOMINAL_DM

def test_analytic_gaussian_exposure_and_fluence():
 dt=.0001;t=np.arange(2001)*dt;freq=np.array([REF_FREQ,REF_FREQ+.0244140625]);q=np.array([0.,.1,.002,0.,0.,0.])
 z=gaussian_model(q,freq,t,ff=1,ft=1);expected=np.exp(-.5*((t-.1)/.002)**2)
 np.testing.assert_allclose(z[0],expected,rtol=1e-12,atol=1e-13);np.testing.assert_allclose(z[1],expected,rtol=1e-12,atol=1e-13)
 integral=gaussian_model(q,freq,t,ff=1,analytic_time=True).sum(1)*dt
 np.testing.assert_allclose(integral,np.sqrt(2*np.pi)*.002,rtol=1e-12)
 assert z[0,0]<z[0,1000] and z[0,-1]<z[0,1000]

def test_native_calibration_and_rms_exact():
 d=load(T/'data/raw_data.npz');assert d['measurements'].dtype==np.float64 and d['measurements'].shape==(16384,162)
 np.testing.assert_array_equal(d['frequencies'],400.2197265625+np.arange(16384)*.0244140625+.0244140625/2)
 np.testing.assert_array_equal(d['times'],np.arange(162)*.0009830400813370943)
 np.testing.assert_array_equal(d['noise_rms'],np.sqrt(np.mean(d['measurements'][:,:50]**2,1)))
 assert d['good_channels'].sum()==10712 and K==4149.377593360996 and NOMINAL_DM==623.6785965856119
