from pathlib import Path
import re
import numpy as np
from src.preprocessing import load_observations,prepare_intensity,HEIGHT,WIDTH,FRAMES
ROOT=Path(__file__).resolve().parents[2]

def test_public_inventory_is_complete_batch_first_and_no_source_urls():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert not re.search(r'github(?:usercontent)?\.com|github\.io|git\.io',text,re.I)
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        assert set(z.files)=={'intensity','polarizer_psg','polarizer_psa'}
        for k in z.files:
            assert z[k].shape[0]==1 and np.isfinite(z[k]).all()
            assert re.search(r'\|\s*`'+k+r'`\s*\|',text)
        assert z['intensity'].shape==(1,FRAMES,HEIGHT,WIDTH)
        assert z['intensity'].dtype==np.float32
        assert z['intensity'].max()>150 and z['intensity'].min()>0
        for k in ('polarizer_psg','polarizer_psa'):
            np.testing.assert_allclose(np.unique(np.rad2deg(z[k])),[0,45,90,135],atol=1e-12)

def test_lower_clipping_preserves_HDR_and_signed_outputs_are_not_inputs():
    a=np.array([[[-1.,.1,155.84]]]);b=prepare_intensity(a)
    np.testing.assert_array_equal(b,[[[0.,.1,155.84]]])
    assert not np.shares_memory(a,b)
