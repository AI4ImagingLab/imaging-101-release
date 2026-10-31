from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_data,prepare
from src.solvers import reconstruct
from src.physics_model import interval_integrals,senum_yang,ratio_objective
T=Path(__file__).resolve().parents[2]
@pytest.fixture(scope='module')
def bound():return prepare(load_data(T/'data/raw_data.npz'))
def test_preprocessing(bound):
    with np.load(T/'test/fixtures/upstream.npz',allow_pickle=False) as a:
        for k,q in [('beta','beta'),('alpha','selected_alpha'),('temperature','selected_T'),('rate','selected_rate'),('advanced_alpha','advanced_alpha'),('advanced_temperature','advanced_T'),('advanced_time','advanced_time')]:np.testing.assert_allclose(bound[k],a[q],rtol=1e-8,atol=1e-9)
def test_full_methods(bound):
    r=reconstruct(bound,True)
    with np.load(T/'test/fixtures/upstream.npz',allow_pickle=False) as a:
        for k in ['Fr','OFW','KAS','Vy','aVy','aVy_temperature']:
            np.testing.assert_allclose(r[k][0],a[k][0],rtol=1e-6,atol=1e-5)
            np.testing.assert_allclose(r[k][1],a[k][1],rtol=1e-8,atol=.01)
def test_forward_original(bound):
    with np.load(T/'test/fixtures/upstream.npz',allow_pickle=False) as a:
        v=np.array([[interval_integrals(e,j,bound) for j in [0,40,80]] for e in a['probe_energy']]);np.testing.assert_allclose(v,a['J_probe'],rtol=1e-8,atol=1e-9)
        for e in a['probe_energy']:
            for j in [0,40,80]:assert ratio_objective(interval_integrals(e,j,bound))>=20-1e-10
        v=np.array([[senum_yang(e,bound['temperature'][j])/bound['beta'] for j in a['probe_rows']] for e in a['probe_energy']])
        np.testing.assert_allclose(v,a['I_probe'],rtol=1e-8,atol=1e-9)
