from pathlib import Path
import numpy as np
from src.preprocessing import DT_PS,load_data
ROOT=Path(__file__).resolve().parents[2]
def test_readme_public_contract_and_constants():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    assert str(DT_PS) in text
    for value in ['5e7','2017-03-20','267','50','float64','waveforms','expected_waveforms','noise_std']:assert value in text
    x=load_data(ROOT/'data/raw_data.npz');assert x.shape==(50,267) and x.dtype==np.float64
    assert [line for line in text.splitlines() if line.startswith('## ')]==['## Background','## Problem Description','## Data Description','## Method Hints','## References']
def test_visual_metric_known_case():
    from src.visualization import relative_error
    assert relative_error(np.array([2.,4.]),np.array([1.,2.]))==1
