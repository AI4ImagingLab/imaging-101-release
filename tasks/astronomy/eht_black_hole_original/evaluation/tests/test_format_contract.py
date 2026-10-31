from pathlib import Path

import numpy as np

from src.preprocessing import load_metadata


TASK_DIR = Path(__file__).resolve().parents[2]


def test_judge_filename_is_lowercase():
    names = {path.name for path in (TASK_DIR / "evaluation").iterdir()}
    assert "judge.py" in names
    assert "Judge.py" not in names


def test_visible_observations_match_readme_table():
    readme = (TASK_DIR / "README.md").read_text(encoding="utf-8")
    with np.load(TASK_DIR / "data" / "raw_data.npz", allow_pickle=False) as raw:
        assert len(raw.files) == 15
        assert "vis_cal" not in raw.files and "image" not in raw.files
        for key in raw.files:
            value = raw[key]
            assert value.shape[0] == 1
            assert f"| `{key}` | `{value.shape}` | {value.dtype} |" in readme


def test_acquisition_constants_match_readme():
    readme = (TASK_DIR / "README.md").read_text(encoding="utf-8")
    meta = load_metadata(str(TASK_DIR / "data"))
    assert meta["N"] == 64 and "| Image grid | 64 x 64 |" in readme
    assert meta["pixel_size_uas"] == 2.0 and "| Pixel pitch | 2.0 |" in readme
    assert f"| Pixel pitch in radians | {meta['pixel_size_rad']} |" in readme
    assert meta["total_flux"] == 0.6 and "| Total source flux | 0.6 |" in readme
    assert meta["freq_ghz"] == 230.0 and "| Observing frequency | 230.0 |" in readme
    assert meta["n_stations"] == 7 and "| Stations | 7 |" in readme
    assert meta["n_baselines"] == 421 and "| Visibilities | 421 |" in readme
