"""Frozen measured-data identity and solver-visible physical contract."""
import hashlib
from pathlib import Path
import re
import numpy as np
from src.physics_model import DIAMETER_CM, WAVELENGTH_ANGSTROM, PIXEL_ARCSEC

ROOT = Path(__file__).resolve().parents[2]


def test_measured_input_is_unchanged_and_every_key_is_publicly_documented():
    readme = (ROOT/'README.md').read_text(encoding='utf-8')
    with np.load(ROOT/'data/raw_data.npz', allow_pickle=False) as raw:
        assert set(raw.files) == {'observations', 'kl_mixing'}
        for key in raw.files:
            assert f'| `{key}` |' in readme
        assert hashlib.sha256(raw['observations'].tobytes()).hexdigest() == '8d96c0ac5983f0dbe9273ec8d82ca7da38749d77f624354dbec3295faec73fb9'


def test_physical_constants_match_acquisition_table_and_public_urls():
    readme = (ROOT/'README.md').read_text(encoding='utf-8')
    assert f'| Telescope aperture diameter | {DIAMETER_CM} | cm |' in readme
    assert f'| Effective model wavelength, both channels | {WAVELENGTH_ANGSTROM} | Angstrom |' in readme
    assert f'| Detector sampling | {PIXEL_ARCSEC} | arcsec/pixel |' in readme
    for path in [ROOT/'README.md', ROOT/'requirements.txt']:
        assert not re.search(r'github\.com|githubusercontent\.com|github\.io|git\.io', path.read_text(encoding='utf-8'), re.I)
