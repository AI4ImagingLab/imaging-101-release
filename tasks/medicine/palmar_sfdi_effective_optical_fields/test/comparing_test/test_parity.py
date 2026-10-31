from pathlib import Path
import numpy as np
from src.preprocessing import load_data,demodulate,average_blocks
from src.physics_model import diffuse_reflectance,objective
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_original_octave_demodulation():
 hand,phantom,rcal,freq,cal,obs=load_data(ROOT/'data/raw_data.npz');a=np.load(ROOT/'test/fixtures/demodulation.npz');np.testing.assert_array_equal(rcal,a['param_phantom_reflectance']);np.testing.assert_array_equal(demodulate(hand[0])[0],a['output_dc_hand']);np.testing.assert_array_equal(demodulate(phantom[0])[0],a['output_dc_phantom']);np.testing.assert_array_equal(cal,a['output_calibrated'])

def test_original_inverse_and_forward():
 hand,phantom,rcal,freq,cal,obs=load_data(ROOT/'data/raw_data.npz');a=np.load(ROOT/'test/fixtures/upstream.npz');np.testing.assert_array_equal(obs,a['output_binned'])
 for row,col in [(0,0),(20,20),(49,39)]:
  np.testing.assert_allclose(diffuse_reflectance(a['output_parameters'][row,col],freq),a['output_predicted'][row,col],rtol=2e-14,atol=1e-16)
 with np.load(ROOT/'output/reconstruction.npz') as fresh:
  np.testing.assert_array_equal(fresh['reconstruction'][0].transpose(1,2,0),a['output_parameters']);np.testing.assert_allclose(fresh['reflectance'][0].transpose(1,2,0),a['output_predicted'],rtol=1e-13,atol=1e-15)

def test_three_phase_analytic_and_block_conservation():
 phases=np.arange(3)*2*np.pi/3;signal=(7+2*np.cos(phases+.37))[:,None,None];dc,ac=demodulate(signal);np.testing.assert_allclose(dc,7,rtol=1e-7);np.testing.assert_allclose(ac,2,rtol=2e-7)
 x=np.arange(120,dtype='float32').reshape(6,10,2);b=average_blocks(x,2);np.testing.assert_allclose(b.mean((0,1)),x.mean((0,1)),rtol=1e-7)

def test_physical_limits_scaling_and_monotonicity():
 f=np.array([0,.1,.2,.3,.4]);a=diffuse_reflectance([.02,1],f);assert np.all(a>0) and np.all(np.diff(a)<0);assert diffuse_reflectance([.04,1],f)[0]<a[0] # Only DC monotonicity is a valid physical check of this approximation.
 # Inverse-length rescaling of frequencies and both coefficients leaves dimensionless response unchanged.
 np.testing.assert_allclose(diffuse_reflectance([.2,10],10*f),a,rtol=1e-14)
 n=1.4;r=-1.440/n**2+.710/n+.668+.0636*n;A=(1-r)/(2*(1+r));np.testing.assert_allclose(diffuse_reflectance([0.,1.],np.array([0.])),[1.],rtol=1e-14)
 np.testing.assert_allclose(diffuse_reflectance([.02,1],np.array([1e9]))[0],0,atol=1e-18)
 high=1e8;np.testing.assert_allclose(diffuse_reflectance([.02,1],np.array([high]))[0]*(2*np.pi*high)**2,3*A*1*1.02,rtol=1e-8)

def test_complete_pipeline_reconstruction():
 *_,freq,cal,obs=load_data(ROOT/'data/raw_data.npz');p,v=reconstruct(obs,freq);a=np.load(ROOT/'output/reconstruction.npz');np.testing.assert_array_equal(p,a['reconstruction'][0].transpose(1,2,0));np.testing.assert_array_equal(v,a['reflectance'][0].transpose(1,2,0));assert objective(p[0,0],freq,obs[0,0])>=0


def test_five_frequency_comparison_preserved():
 *_,freq,cal,obs=load_data(ROOT/'data/raw_data.npz');p,v=reconstruct(obs,freq,frequency_indices=(0,1,2,3,4));a=np.load(ROOT/'test/fixtures/upstream_five_frequency.npz');np.testing.assert_array_equal(p,a['output_parameters']);np.testing.assert_allclose(v,a['output_predicted'],rtol=1e-13,atol=1e-15)
