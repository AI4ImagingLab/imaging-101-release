from pathlib import Path
import numpy as np
import pytest
from src.main import reconstruct
T=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('name,method,directions,p1,p2',[
    ('sgm','sgm',8,8,32),('ocsgm','ocsgm',8,8,32),('mgm4','mgm',4,8,32),
    ('mgm_soft','mgm',8,4,16),('mgm_strong','mgm',8,16,64)])
def test_published_comparison_groups_against_independent_original(name,method,directions,p1,p2):
    fresh,_=reconstruct(T/'data/raw_data.npz',method,directions,p1,p2)
    with np.load(T/'notebooks/reference_outputs/upstream_comparisons.npz',allow_pickle=False) as data:
        np.testing.assert_allclose(fresh['disparity'][0],data[name],atol=1e-5,rtol=0)
