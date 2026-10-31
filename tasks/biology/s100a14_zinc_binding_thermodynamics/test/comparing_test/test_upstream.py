from pathlib import Path
import numpy as np
from src.physics_model import concentrations, predict
from src.preprocessing import load_data, ACQUISITION
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
F=np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)
def test_constants_and_dilution():
    data=load_data(ROOT/'data/raw_data.npz')
    for i,a in enumerate(ACQUISITION):
        np.testing.assert_array_equal(a,F[f'config_acquisition_{i}'])
        np.testing.assert_array_equal(data['heats'][i],F[f'input_all_heats_{i}'])
        s,t=concentrations(data['injection_volumes'][i],a[1],a[2],a[3])
        np.testing.assert_allclose(s,F[f'output_S_{i}'],rtol=1e-13,atol=0)
        np.testing.assert_allclose(t,F[f'output_T_{i}'],rtol=1e-13,atol=0)
def test_full_forward_stage_parity():
    data=load_data(ROOT/'data/raw_data.npz'); q=predict(F['output_parameters'],data['injection_volumes'])
    for i in range(6):
        np.testing.assert_allclose(q[i],F[f'output_dQ_{i}'],rtol=1e-11,atol=1e-10)
def test_full_inverse_parity():
    data=load_data(ROOT/'data/raw_data.npz'); p=reconstruct(data).x
    # Frozen before extraction from 5 original multistarts; nuisance compensation dominates.
    limits=np.array([.2,.5,2e-5]+[2.,.002]*6)
    assert np.all(np.abs(p-F['output_parameters'])<=limits)
