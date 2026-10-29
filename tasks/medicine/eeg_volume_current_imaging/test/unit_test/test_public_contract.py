from pathlib import Path
import re
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def test_public_keys_shapes_units_and_acquisition_constants():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as data:
        for key in data.files:
            rows=[r for r in text.splitlines() if r.startswith('| '+key+' |')]
            assert rows, key
            cells=[c.strip() for c in rows[-1].split('|')[1:-1]]
            assert len(cells)==4
            assert tuple(map(int,re.findall(r'\d+',cells[1])))==data[key].shape
            assert cells[2]==str(data[key].dtype)
            assert cells[3]
    for value in ['160', '9.99375', '0.015', '0.065', '0.09', '0.004']: assert value in text
    assert not re.search(r'github(?:usercontent)?\.com|github\.io|git\.io',text,re.I)
