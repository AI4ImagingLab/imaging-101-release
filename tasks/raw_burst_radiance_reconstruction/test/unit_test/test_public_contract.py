from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def test_public_data_and_metadata():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:
        assert f.files==['frames'];assert f['frames'].shape==(1,10,1024,1024);assert f['frames'].dtype==np.uint16
    for token in ['3.55148388','516.6520187906699','1023','1143','1048:2072','1592:2616','BGGR','CC BY-SA 4.0']:assert token in text
    assert 'github' not in text.lower()
