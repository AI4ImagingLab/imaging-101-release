from pathlib import Path
import numpy as np
T=Path(__file__).resolve().parents[2]
def test_public_allowlist_and_metadata():
    text=(T/'README.md').read_text(encoding='utf8')
    for domain in ['github.com','github.io','githubusercontent.com','git.io']: assert domain not in text.lower()
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as z:
        assert set(z.files)=={'edge_indices','relative_poses','information'}
        for key in z.files: assert '| '+key+' |' in text and z[key].shape[0]==1
        assert z['edge_indices'].max()==1227 and z['edge_indices'].dtype==np.int64
        assert z['relative_poses'].shape==(1,1483,3)
    assert '1228' in text and '1483' in text and '(-pi,pi]' in text
    assert not (T/'meta_data.json').exists()
