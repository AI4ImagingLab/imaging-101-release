
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
def test_public_analysis_band_and_output_grid():
    from src.solvers import prepare
    r=metadata();d=load_data(ROOT/'data/raw_data.npz');f,h,dt=prepare(d['frequency'],d['accelerance']);assert r['Analysis band']=='10 to 5000';assert len(f)==4999 and f[0]==0 and f[-1]<5000;assert d['accelerance'].shape==(6,9999);assert r['Time convention']=='Laplace s=i omega'
