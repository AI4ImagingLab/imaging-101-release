import numpy as np
from pathlib import Path
from src.preprocessing import load_data,shell_average,signal_mask
from src.physics_model import design_matrix
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_original_stages_and_pipeline():
    data=load_data(ROOT/'data/raw_data.npz');f=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    means,counts,b=shell_average(data['signals'][0],data['bvalues'][0])
    np.testing.assert_array_equal(means,f['output_shell_means']);np.testing.assert_array_equal(counts,f['output_counts'])
    np.testing.assert_array_equal(design_matrix(b),f['output_design']);np.testing.assert_array_equal(signal_mask(data['signals'][0],data['bvalues'][0]),f['input_mask'])
    out=reconstruct(data);ref=np.load(ROOT/'evaluation/baseline_reference.npz')
    for k in out:np.testing.assert_allclose(out[k],ref[k],rtol=2e-9,atol=1e-10)
