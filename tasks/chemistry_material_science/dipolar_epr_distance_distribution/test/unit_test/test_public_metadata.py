
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
def test_public_physical_constants_and_probability_grid():
    from src import physics_model as model
    r=metadata()
    for key,name in [('mu0','MU0'),('Bohr magneton','MUB'),('Electron g','G'),('Planck constant','H'),('Avogadro constant','NAV')]:assert float(r[key])==getattr(model,name)
    np.testing.assert_array_equal(model.DISTANCE,np.arange(1.5,6,.01));assert float(r['First interpulse delay'])==.3 and float(r['Second interpulse delay'])==4. and float(r['Pulse length'])==.016
