"""Exact bound observations, frozen upstream stages, and fresh full pipeline."""
from pathlib import Path
import numpy as np
import torch
from src.preprocessing import load_observations,prepare_matrix
from src.solvers import solve
from src.main import run

ROOT=Path(__file__).resolve().parents[2]


def test_bound_preprocessing_upstream_parity():
    matrix,_=prepare_matrix(load_observations(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/preprocessing.npz',allow_pickle=False) as fixture:
        np.testing.assert_allclose(matrix[:,[0,13,149]],fixture['output_R_columns'],rtol=2e-6,atol=.02)


def test_upstream_three_stage_updates():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    _,details=solve(load_observations(ROOT/'data/raw_data.npz'),iterations=3,device=device,record_steps=(1,2,3))
    with np.load(ROOT/'test/fixtures/updates.npz',allow_pickle=False) as fixture:
        for index in (1,2,3):
            actual=details['snapshots'][index]
            expected=fixture['output_phase_'+str(index)]
            assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<2e-4


def test_full_fresh_pipeline_against_corrected_upstream(tmp_path):
    result,details=run(ROOT/'data/raw_data.npz',tmp_path/'nested/reconstruction.npz')
    with np.load(ROOT/'notebooks/reference_outputs/upstream_paper_corrected.npz',allow_pickle=False) as reference:
        for key in ['reconstruction','mtf']:
            np.testing.assert_allclose(result[key],reference[key],rtol=2e-4,atol=2e-5)
    # Compare the measured independent upstream value, not an invented gain.
    assert abs(details['final_coherence']-7.270196452047658)<2e-5
    assert details['final_coherence']>details['initial_coherence']
    assert details['iterations']==250
