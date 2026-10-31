from pathlib import Path
import numpy as np
from src.preprocessing import preprocess,load_data
from src.physics_model import decoding_pattern,central_crop,projected_size
from src.solvers import reconstruct,accorsi
ROOT=Path(__file__).resolve().parents[2]
def test_all_original_stages():
    y,m=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(preprocess(y),z['output_preprocessed'])
        np.testing.assert_array_equal(decoding_pattern(m),z['output_decoder'])
        np.testing.assert_array_equal(central_crop(preprocess(y),projected_size()/256),z['output_crop'])
        np.testing.assert_allclose(reconstruct(y,m),z['output_mura'],rtol=2e-6,atol=.05)
        np.testing.assert_allclose(accorsi(y,m),z['output_accorsi'],rtol=2e-6,atol=.02)
