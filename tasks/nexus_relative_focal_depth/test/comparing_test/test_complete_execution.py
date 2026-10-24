"""Inspect mandatory full-run measurements, not a third redundant reconstruction.

Independent original and fresh CLI/activeNotebook runs are separately timed in
validation; missing recorded full-run assets are failures, never cache fallbacks.
"""
from pathlib import Path
import json,importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def test_full_native_reconstruction_against_original():
    spec=importlib.util.spec_from_file_location('full_judge',ROOT/'evaluation/judge.py')
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz')
    assert code==0,report


def test_all_executed_stage_measurements():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as expected:
        with np.load(ROOT/'notebooks/reference_outputs/fresh_cli_stage_measurements.npz',allow_pickle=False) as actual:
            np.testing.assert_allclose(actual['raw_sharpness'],expected['raw_sharpness'],atol=2e-6,rtol=0)
            np.testing.assert_allclose(actual['positive_focus_coefficients'],expected['positive_focus_coefficients'],atol=2e-6,rtol=0)
            np.testing.assert_allclose(actual['negative_focus_derivative'],expected['negative_focus_derivative'],atol=1e-6,rtol=0)
            np.testing.assert_allclose(actual['initial_focal_coordinate'],expected['initial_focal_coordinate'],atol=1e-4,rtol=0)
            for n in [1,2,10,100,400]:
                # Intermediate endpoints use max source FMA-variant trajectory
                # spread, fixed before extraction; full grid has tighter Judge.
                np.testing.assert_allclose(actual[f'step{n}_u'],expected[f'step{n}_u'],atol=.016,rtol=0)
                for variable in ['gx','gy','bx','by','ux','uy']:
                    np.testing.assert_allclose(actual[f'step{n}_{variable}'],expected[f'step{n}_{variable}'],atol=.016,rtol=0)


def test_completed_notebook():
    path=ROOT/'notebooks/nexus_relative_focal_depth.ipynb'
    notebook=json.loads(path.read_text(encoding='utf8'))
    code=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert len(code)>=5 and all(c.get('execution_count') is not None for c in code)
    assert not [o for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error']
    text='\n'.join(''.join(c['source']).lower() for c in notebook['cells'] if c['cell_type']=='markdown')
    assert all(word in text for word in ['problem','data','physics','method','results','conclusion'])
