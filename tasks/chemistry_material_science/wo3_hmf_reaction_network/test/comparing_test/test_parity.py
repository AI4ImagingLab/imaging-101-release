from pathlib import Path
import numpy as np
from src.physics_model import concentrations,charge
from src.preprocessing import load_data,preprocess
from src.solvers import residual,reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_upstream_stages():
    f=np.load(ROOT/'test/fixtures/upstream.npz')
    c,q,c0=preprocess(load_data(ROOT/'data/raw_data.npz'))
    np.testing.assert_array_equal(c,f['mean_concentrations'])
    np.testing.assert_array_equal(q,f['mean_charge'])
    curves=concentrations(f['time'],f['c0'],f['rates'])
    np.testing.assert_allclose(curves,f['concentrations'],rtol=2e-9,atol=1e-8)
    np.testing.assert_allclose(charge(curves),f['charge'],rtol=2e-9,atol=1e-9)
    np.testing.assert_allclose(residual(np.full(11,.05),c,q,c0),f['initial_residual'],rtol=2e-9,atol=1e-10)

def test_complete_fit():
    c,q,c0=preprocess(load_data(ROOT/'data/raw_data.npz'))
    out,info=reconstruct(c,q,c0)
    f=np.load(ROOT/'test/fixtures/upstream.npz')
    assert abs(info['objective']-4.949217144795996)<1e-5
    assert info['objective']<7.504028595084471
    assert np.all(abs(out['rates'][0]-f['rates'])<=2e-5+.002*abs(f['rates']))
    for j in range(15):
        e=np.sqrt(np.mean((out['reconstruction'][0,:,j]-f['concentrations'][:,j])**2))/np.ptp(f['concentrations'][:,j])
        assert e<.002

def test_original_parsed_observation_hashes():
    import json,hashlib
    expected=json.loads((ROOT/'test/fixtures/author_parsed_hashes.json').read_text())
    with np.load(ROOT/'data/raw_data.npz') as f:
        for key,item in expected.items():
            assert list(f[key].shape)==item['shape'] and str(f[key].dtype)==item['dtype']
            assert hashlib.sha256(f[key].tobytes()).hexdigest()==item['sha256']
