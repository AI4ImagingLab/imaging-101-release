from pathlib import Path
import importlib.util
import numpy as np
from src.preprocessing import load_data, crop_slices, PHASE_PER_PPM

TASK = Path(__file__).resolve().parents[2]


def test_real_observation_contract_and_metadata():
    obs = load_data(TASK/'data/raw_data.npz')
    assert obs['mask'].sum() == 933481
    assert obs['phase'].shape == (160, 160, 160)
    assert crop_slices(obs['mask']) == (slice(18, 143), slice(7, 157), slice(11, 137))
    assert np.max(abs(obs['phase'])) <= np.pi+1e-6
    text = (TASK/'README.md').read_text(encoding='utf-8')
    for value in ['1.0625', '1.0714285714285714', '2.89362', '0.025', '42.58']:
        assert value in text
    assert abs(PHASE_PER_PPM-19.35383488668318) < 1e-12


def test_fresh_standard_entry_point_output_against_standalone_evaluator():
    # The full CLI is run and logged independently before the test suite.
    # This consumer validates its fresh artifact without a second costly solve.
    spec=importlib.util.spec_from_file_location('qsm_pipeline_judge',TASK/'evaluation/judge.py')
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    result,code=judge.evaluate(TASK/'output/reconstruction.npz')
    assert code == 0, result
