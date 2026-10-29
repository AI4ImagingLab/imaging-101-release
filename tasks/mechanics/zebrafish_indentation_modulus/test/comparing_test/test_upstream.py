from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import preprocess,load_data
from src.physics_model import contact_force,residual
from src.solvers import fit_curve,reconstruct
T=Path(__file__).resolve().parents[2]
F=np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False)
@pytest.mark.parametrize('i',[0,1,12,100,500,900,1100])
def test_frozen_original_stages(i):
    f,h,s,k=[F[f'{p}_{i}'] for p in ['input_force','input_height','input_segment','param_spring']]
    x,y,o=preprocess(f,h,k)
    np.testing.assert_allclose(x,F[f'output_tip_{i}'],rtol=0,atol=1e-18)
    np.testing.assert_allclose(y,F[f'output_force_{i}'],rtol=0,atol=1e-22)
    E,C,_=fit_curve(f,h,s,k);mask=s==0
    np.testing.assert_allclose(E,F[f'output_E_{i}'],rtol=2e-4,atol=2e-4)
    np.testing.assert_allclose(C,F[f'output_contact_{i}'],rtol=0,atol=2e-9)
    expected_E=F[f'output_E_{i}']; expected_C=F[f'output_contact_{i}']-o
    np.testing.assert_allclose(contact_force(x[mask],expected_E,expected_C),F[f'output_prediction_{i}'][mask],rtol=2e-12,atol=1e-22)
    np.testing.assert_allclose(residual((expected_E,expected_C),x[mask],y[mask]),F[f'output_residual_{i}'][mask],rtol=2e-12,atol=1e-22)

def test_full_pipeline():
    p=reconstruct(load_data(T/'data/raw_data.npz'))
    with np.load(T/'evaluation/baseline_reference.npz') as r:
        np.testing.assert_allclose(p['reconstruction'],r['reconstruction'],rtol=2e-4,atol=2e-4)
        np.testing.assert_allclose(p['contact_position'],r['contact_position'],rtol=0,atol=2e-9)
