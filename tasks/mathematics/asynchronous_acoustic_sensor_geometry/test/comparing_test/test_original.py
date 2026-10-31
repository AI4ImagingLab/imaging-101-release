from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import pack,unpack,residual_jacobian
from src.solvers import normal_step,reconstruct
ROOT=Path(__file__).resolve().parents[2]

def inputs():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:fixture={k:f[k] for k in f.files}
    return data,fixture

def test_original_initial_and_final_stages():
    data,f=inputs()
    for label in ['initial','final']:
        x=f[label];residual,jacobian=residual_jacobian(x,data)
        np.testing.assert_allclose(residual,f['residual_'+label],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(jacobian,f['J_'+label],rtol=1e-8,atol=1e-9)
        np.testing.assert_array_equal(unpack(pack(x)),x)
    step=normal_step(f['J_initial'],f['residual_initial'])
    np.testing.assert_allclose(step,f['first_step'],rtol=1e-6,atol=1e-6)

def test_original_complete_geometry_and_objective():
    data,f=inputs();result,diagnostic=reconstruct(data)
    np.testing.assert_allclose(result['reconstruction'][0],f['final'][:,:3],rtol=0,atol=.002)
    assert abs(diagnostic['cost']-201.653969730)<.01
    assert diagnostic['successful_starts']>0
