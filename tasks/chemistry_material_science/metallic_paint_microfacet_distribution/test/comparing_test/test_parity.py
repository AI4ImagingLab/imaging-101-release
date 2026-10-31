from pathlib import Path
import numpy as np
from src.preprocessing import load_data,backscatter,LUMINANCE
from src.physics_model import build_kernel,mass,roughness,eigen_residual
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_full_independent_upstream_stages():
    f=np.load(ROOT/'test/fixtures/upstream.npz'); result,d=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    np.testing.assert_allclose(np.c_[d['theta'],d['rgb']],f['retro'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(d['kernel'],f['kernel'],atol=1e-9,rtol=1e-10)
    np.testing.assert_allclose(d['powers'],f['powers'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(result['reconstruction'][0],f['pdf'],atol=1e-8,rtol=0)
    np.testing.assert_allclose(result['roughness'][0],f['roughness'],atol=1e-9,rtol=0)
def test_physical_mass_positivity_and_direct_eigen_solution():
    result,d=reconstruct(load_data(ROOT/'data/raw_data.npz')); p=result['reconstruction'][0]
    assert abs(mass(p)-1)<1e-10 and np.all(p>=0) and p[-1]==0
    values,vectors=np.linalg.eig(d['kernel']); v=np.real(vectors[:,np.argmax(values.real)])
    v=np.r_[v,0.]; v/=mass(v)
    assert np.linalg.norm(v-p)/np.linalg.norm(v)<1e-5
    assert eigen_residual(d['kernel'],p)[0]<1e-5
def test_acquisition_intensity_scale_invariance():
    data=load_data(ROOT/'data/raw_data.npz'); a,_=reconstruct(data); b,_=reconstruct(data*3)
    np.testing.assert_allclose(a['reconstruction'],b['reconstruction'],rtol=1e-12)
    np.testing.assert_allclose(a['roughness'],b['roughness'],rtol=1e-12)
def test_independent_finite_weights_and_moments():
    import math
    result,_=reconstruct(load_data(ROOT/'data/raw_data.npz')); p=result['reconstruction'][0]
    total=second=first=0.
    for k in range(128):
        t=k/128*math.sqrt(math.pi/2); angle=t*t; slope=math.tan(angle)
        coordinate=math.sqrt(2*math.atan(slope)/math.pi)*89
        lo=int(coordinate); fraction=coordinate-lo
        value=p[lo]*(1-fraction)+p[min(lo+1,89)]*fraction
        weight=2*math.sqrt(math.pi/2)/128*t*value*slope/math.cos(angle)**2
        assert weight>=0
        total+=2*math.pi*weight; second+=2*math.pi*slope*slope*weight; first+=4*slope*weight
    assert abs(total-1)<1e-10
    np.testing.assert_allclose(result['roughness'][0],[math.sqrt(second),first],rtol=1e-12)
    assert p[-2]>0  # Explicit evidence that infinite-slope interpolation has a nonintegrable tail.
