from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import calibrated_phasors,mixture_phasors
from src.solvers import reconstruct,graphical_search
T=Path(__file__).resolve().parents[2]
def test_calibrated_original():
    signal,reference=load_data(T/'data/raw_data.npz');m,z=calibrated_phasors(signal,reference)
    with np.load(T/'test/fixtures/upstream_fields.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(m,f['mean'],atol=1e-10,rtol=1e-9)
        np.testing.assert_allclose(z.real,f['real'],atol=1e-10,rtol=1e-9,equal_nan=True);np.testing.assert_allclose(z.imag,f['imag'],atol=1e-10,rtol=1e-9,equal_nan=True)
def test_full_original():
    s,r=load_data(T/'data/raw_data.npz');out,details=reconstruct(s,r)
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        for k in f.files:np.testing.assert_allclose(out[k],f[k],atol=1e-5,rtol=1e-6)
    with np.load(T/'test/fixtures/upstream_fields.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(details['single_fraction'],np.nan_to_num(f['single_fraction']),atol=1e-10,rtol=1e-9)
        np.testing.assert_allclose(out['reconstruction'][0],np.nan_to_num(f['lifetimes']),atol=1e-5,rtol=1e-6)
        np.testing.assert_allclose(details['fractions'],f['fractions'],atol=1e-10,rtol=1e-9,equal_nan=True)

def test_frozen_candidate_and_single_endpoint_stages():
    from src.physics_model import OMEGA
    from src.solvers import single_harmonic
    with np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as stage:
        candidate=stage['phasor_from_lifetime_45_input1']
        np.testing.assert_array_equal(candidate,np.arange(.2,.8,.01))
        formula=1/(1-1j*OMEGA*candidate)
        np.testing.assert_allclose(formula.real,stage['phasor_from_lifetime_45_output0'],atol=1e-14)
        np.testing.assert_allclose(formula.imag,stage['phasor_from_lifetime_45_output1'],atol=1e-14)
        with np.load(T/'test/fixtures/upstream_fields.npz',allow_pickle=False) as f:
            z=f['real'][0]+1j*f['imag'][0];m=f['mean'];valid=np.isfinite(z)
            center=np.nansum(z*m)/m[valid].sum()
            np.testing.assert_allclose(center,stage['phasor_center_1_output1']+1j*stage['phasor_center_1_output2'],atol=1e-14)
            _,bound=single_harmonic(z,m)
            np.testing.assert_allclose(bound,stage['phasor_to_normal_lifetime_1_output0'],atol=1e-13)
            endpoint=1/(1-1j*OMEGA*bound)
            np.testing.assert_allclose(endpoint,stage['phasor_semicircle_intersect_1_output2']+1j*stage['phasor_semicircle_intersect_1_output3'],atol=1e-14)
def test_independent_known_mixture():
    tau=np.array([.4,3.4]);fraction=.6;z=mixture_phasors(tau[:,None],np.array([fraction,1-fraction])[:,None])
    t,f=graphical_search(z)
    np.testing.assert_allclose(t[:,0],tau,atol=1e-8)
    np.testing.assert_allclose(f[:,0],[fraction,1-fraction],atol=1e-8)
