
from pathlib import Path
import numpy as np
from src.preprocessing import load_data
ROOT=Path(__file__).resolve().parents[2]
def metadata():
    text=(ROOT/'README.md').read_text(encoding='utf8').split('**Acquisition metadata**')[1].split('## Method Hints')[0]
    rows={}
    for line in text.splitlines():
        cells=[v.strip() for v in line.split('|')]
        if len(cells)==6 and cells[1] not in ['Parameter','---']:rows[cells[1]]=cells[2]
    return rows
def test_public_modulus_frequency_and_output_conventions():
    from src.solvers import ReSpectConfig
    from src.preprocessing import prepare
    r=metadata();config=ReSpectConfig();d=load_data(ROOT/'data/raw_data.npz');w,g,weights=prepare(d);assert r['Frequency convention']=='Angular frequency';assert float(r['Fluid plateau'])==0 and not config.plateau;assert config.ns==100 and 'j=0..99' in r['Continuous output grid'];assert len(w)==100 and len(g)==200;assert (g>0).all() and np.array_equal(weights,np.ones(200))
