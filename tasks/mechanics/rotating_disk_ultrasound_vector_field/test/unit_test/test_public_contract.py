from pathlib import Path
import importlib
import numpy as np
from src import physics_model as physics

ROOT=Path(__file__).resolve().parents[2]
def test_source_instrument_constants_match_public_metadata():
    rows={}
    for line in (ROOT/'README.md').read_text(encoding='utf8').splitlines():
        if line.startswith('| '):
            parts=[p.strip() for p in line.split('|')[1:-1]]
            if len(parts)==4:rows[parts[0]]=parts[1]
    names={'c':('Medium sound speed',physics.SOUND_SPEED),'fc':('Carrier frequency',physics.CARRIER_FREQUENCY),'fs':('Saved RF sampling rate',physics.SAMPLING_FREQUENCY),'t0':('First fast-time sample',physics.START_TIME),'pitch':('Element pitch',physics.PITCH),'width':('Element width',physics.ELEMENT_WIDTH),'bandwidth':('Fractional bandwidth',physics.BANDWIDTH_PERCENT),'PRF':('Pulse repetition frequency',physics.PULSE_REPETITION_FREQUENCY)}
    with np.load(ROOT/'test/fixtures/instrument_parameters.npz',allow_pickle=False) as source:
        for key,(name,value) in names.items():
            assert value==float(source[key][0])
            assert float(rows[name].replace(',',''))==value

def test_main_import_does_not_change_matplotlib_backend():
    import matplotlib
    before=matplotlib.get_backend()
    importlib.import_module('src.main')
    assert matplotlib.get_backend()==before
