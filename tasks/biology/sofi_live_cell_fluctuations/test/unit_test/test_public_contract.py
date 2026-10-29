from pathlib import Path
import re,numpy as np
ROOT=Path(__file__).resolve().parents[2]
def test_readme_and_public_allowlist():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for title in ['Background','Problem Description','Data Description','Method Hints','References']:assert '## '+title in text
    assert 'output/reconstruction.npz' in text and 'Acquisition metadata' in text
    assert not re.search(r'github\.com|githubusercontent|github\.io|git\.io',text,re.I)
    assert not (ROOT/'meta_data.json').exists()
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as d:
        for key in d.files:assert '| '+key+' |' in text and d[key].shape[0]==1

def test_documented_input_shapes_and_storage_dtypes():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as d:
        for key in d.files:
            line=next(x for x in text.splitlines() if x.startswith('| '+key+' |'))
            assert str(d[key].shape) in line and str(d[key].dtype) in line
