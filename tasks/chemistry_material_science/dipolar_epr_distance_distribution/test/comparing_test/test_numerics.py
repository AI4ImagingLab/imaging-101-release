
from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_full_upstream_parity():
    result=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        for k in f.files:
            assert result[k].shape==f[k].shape
            assert np.linalg.norm(result[k]-f[k])/np.linalg.norm(f[k])<1e-5,k

def test_phase_and_powder_kernel_upstream():
    from src.preprocessing import prepare
    from src.physics_model import powder_kernel
    data=load_data(ROOT/'data/raw_data.npz');t,y=prepare(data['time'],data['echo']);f=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    np.testing.assert_allclose(y,f['output_phased'],rtol=1e-13,atol=1e-14);np.testing.assert_allclose(powder_kernel(t),f['output_kernel'],rtol=1e-12,atol=1e-14)
    from src.physics_model import DISTANCE
    np.testing.assert_array_equal(t,f['input_time']);np.testing.assert_array_equal(DISTANCE,f['param_distance']);native=f['output_parameters'];canonical=native[[1,0,2,4,3,5,6,8,7]] if native[3]>native[4] else native
    np.testing.assert_allclose(reconstruct(data)['parameters'][0],canonical,rtol=1e-5)
def test_powder_independent_orientation_integral_and_probability():
    from scipy.integrate import quad
    from src.physics_model import powder_kernel,D,gaussian_basis,DISTANCE
    for t,r in [(0,3.),(.2,4.),(1.1,2.7)]:
        expected=quad(lambda u:np.cos(t*D*1e21/r**3*(1-3*u*u)),0,1,epsabs=1e-12)[0]
        np.testing.assert_allclose(powder_kernel(np.array([t]),np.array([r]),False)[0,0],expected,atol=1e-12)
    result=reconstruct(load_data(ROOT/'data/raw_data.npz'));assert result['reconstruction'].min()>=0;np.testing.assert_allclose(np.trapezoid(result['reconstruction'][0],DISTANCE),1)
