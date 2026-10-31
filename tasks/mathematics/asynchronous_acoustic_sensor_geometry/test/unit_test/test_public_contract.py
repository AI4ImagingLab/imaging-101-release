from pathlib import Path
import numpy as np
from src.physics_model import SOUND_SPEED,MICROPHONES,EVENTS
from src.preprocessing import load_data

def test_public_acquisition_matches_executable_constants():
    root=Path(__file__).resolve().parents[2]
    text=(root/'README.md').read_text(encoding='utf8')
    assert f'| Sound speed | {SOUND_SPEED:g} | m/s |' in text
    assert f'| Microphones / events | {MICROPHONES} / {EVENTS} |' in text
    data=load_data(root/'data/raw_data.npz')
    assert data['tdoa_successive'].shape==(EVENTS-1,MICROPHONES)
    assert data['tdoa_between'].shape==(MICROPHONES-1,EVENTS)
    for key in data:assert f'| {key} |' in text
    assert np.all(np.diff(np.cumsum(data['emission_intervals']))>0)
