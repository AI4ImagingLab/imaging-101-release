
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
def test_public_acquisition_constants_and_grid():
    from src.physics_model import SAMPLE_FREQUENCY,SOUND_SPEED,grid_positions
    from src.preprocessing import INVALID_CHANNELS
    r=metadata();assert float(r['Sample frequency'])==SAMPLE_FREQUENCY;assert float(r['Sound speed'])==SOUND_SPEED;d=load_data(ROOT/'data/raw_data.npz');assert int(r['Samples'])==len(d['microphone_signal']);assert INVALID_CHANNELS==[1,7];g=grid_positions();assert g.shape==(169,3);np.testing.assert_array_equal(np.unique(g[:,0]),np.linspace(-.6,0,13));np.testing.assert_array_equal(np.unique(g[:,1]),np.linspace(-.3,.3,13));assert np.all(g[:,2]==float(r['Source-plane z']))
