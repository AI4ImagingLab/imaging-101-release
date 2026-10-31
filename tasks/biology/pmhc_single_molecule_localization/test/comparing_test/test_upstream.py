from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,bandpass_filter
from src.solvers import reconstruct,find_peaks_yx,locate
ROOT=Path(__file__).resolve().parents[2]

def test_complete_independent_original_stages():
    raw=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        for i,frame in enumerate(raw[0]):
            image=bandpass_filter(frame)
            np.testing.assert_allclose(image,z[f'output_filtered_{i}'],rtol=1e-12,atol=1e-10)
            np.testing.assert_array_equal(find_peaks_yx(image,3,300)[:,::-1],z[f'output_peaks_{i}'])
            np.testing.assert_allclose(locate(frame,3,300,5000),z[f'output_localizations_{i}'],rtol=1e-12,atol=1e-10)
    out=reconstruct(raw)
    np.testing.assert_allclose(out,np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'],rtol=1e-12,atol=1e-10)
