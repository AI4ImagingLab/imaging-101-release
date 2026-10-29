from pathlib import Path
import inspect,re
import numpy as np
from src.preprocessing import load_observations,CONFIDENCE_RATIO,CONFIDENCE_HISTOGRAM_BINS
from src.solvers import maximum_likelihood
from src.physics_model import Q_GRID,P_GRID
ROOT=Path(__file__).resolve().parents[2]

def test_public_array_inventory_batch_shapes_units_and_no_source_URL():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert not re.search(r'github(?:usercontent)?\.com|github\.io|git\.io|codeload\.',text,re.I)
    assert all('## '+s in text for s in ['Background','Problem Description','Data Description','Method Hints','References'])
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        assert len(z.files)==9
        for key in z.files:
            assert f'| `{key}` |' in text
            assert z[key].shape[0]==1 and z[key].dtype.kind in 'ifu' and np.isfinite(z[key]).all()
    assert not (ROOT/'src/generate_data.py').exists()

def test_native_constants_and_output_coordinates_match_declared_contract():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    data=load_observations()
    assert CONFIDENCE_RATIO==.8 and CONFIDENCE_HISTOGRAM_BINS==201
    np.testing.assert_array_equal(data['phase_step'],np.full(100,np.pi/32))
    assert np.array_equal(Q_GRID,np.linspace(-7,7,201)) and np.array_equal(P_GRID,Q_GRID)
    assert '(1, 31, 31)' in text and '(1, 201, 201)' in text
    for value in ('5554.373134328358','5811.4427860696524','7281.3880597014922'):assert value in text
    assert inspect.signature(maximum_likelihood).parameters['iterations'].default==100
    assert inspect.signature(maximum_likelihood).parameters['cutoff'].default==30
