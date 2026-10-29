from pathlib import Path
import numpy as np
from src.physics_model import radial_frequency, sideband_wave
from src.solvers import reconstruct, estimate_carrier
from src.preprocessing import load_observations, PIXEL_SIZE_NM

ROOT = Path(__file__).resolve().parents[2]


def test_original_stage_fixtures():
    data = load_observations(ROOT / 'data/raw_data.npz')
    with np.load(ROOT / 'test/fixtures/upstream.npz', allow_pickle=False) as f:
        np.testing.assert_allclose(radial_frequency((512, 512)), f['frequencies'], atol=1e-15)
        position, radius = estimate_carrier(data['vacuum'][0])
        np.testing.assert_array_equal(position, f['position'])
        assert abs(radius - float(f['radius'])) < 1e-12
        for key, target in [('hologram', 'object_wave'), ('vacuum', 'vacuum_wave')]:
            np.testing.assert_allclose(sideband_wave(data[key][0], position, radius), f[target], rtol=1e-11, atol=1e-10)
        result = reconstruct(data['hologram'], data['vacuum'])
        expected = (f['object_wave'] / f['vacuum_wave'])[None]
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-11)


def test_readme_constants():
    assert str(PIXEL_SIZE_NM) in (ROOT / 'README.md').read_text(encoding='utf8')
