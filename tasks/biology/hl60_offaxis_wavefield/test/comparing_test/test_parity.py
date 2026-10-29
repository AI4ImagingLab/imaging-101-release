from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_original_hologram_empty_fields_and_full_result():
    result,stage=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:np.testing.assert_allclose(result['reconstruction'],f['reconstruction'],rtol=1e-12,atol=1e-12)
    with np.load(ROOT/'test/fixtures/oracle_stages.npz') as f:
        for key in ['object_field','background_field','fft_origin','fft_filtered']:np.testing.assert_allclose(stage[key],f['output_'+key][0],rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(stage['phase_difference'],f['output_comparison_phase_difference'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(stage['unwrapped'],f['output_comparison_unwrapped'],rtol=1e-12,atol=1e-12)
