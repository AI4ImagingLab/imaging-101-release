
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
def test_public_frequency_extent_and_grid_convention():
    from src.physics_model import operators
    d=load_data(ROOT/'data/raw_data.npz');a,l,t=operators(d['frequency']);r=metadata();lo,hi=map(float,r['Acquisition extent'].split(' to '));assert d['frequency'].min()==lo and d['frequency'].max()==hi;np.testing.assert_array_equal(t,1/d['frequency']);assert 'tau[j] = 1 / frequency[j]'==r['Relaxation grid']
