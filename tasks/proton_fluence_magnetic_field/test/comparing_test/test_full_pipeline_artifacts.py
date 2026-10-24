"""Audit complete actual CLI/active-Notebook measurements against original fixtures.

Full400000-update executions are separately mandatory and recorded in validation.
These tests avoid redundantly repeating a25-minute inverse for each test call.
The notebook actively regenerates its fresh measurement; absent assets fail.
"""
from pathlib import Path
import numpy as np
import subprocess,sys,json
from src.physics_model import velocity_from_mapping
from src.preprocessing import load_observations

ROOT=Path(__file__).resolve().parents[2]

def test_complete_executed_state_matches_independent_full_source():
    with np.load(ROOT/'notebooks/reference_outputs/fresh.npz',allow_pickle=False) as fresh, np.load(ROOT/'test/fixtures/upstream_full.npz',allow_pickle=False) as original:
        assert int(fresh['iterations'])==400000
        assert fresh['residual_norm'].shape==(400000,)
        assert int(fresh['negative_determinant_count'])==0
        for name in ['phin','phix','phiy','phixx','phiyy','phixy','fluxn','det','Fn','RMSfluxerror']:
            reference=original[name];scale=max(float(np.max(np.abs(reference))),1e-30)
            assert np.max(np.abs(fresh[name]-reference))/scale<=2e-8
        assert np.min(fresh['det'])>0

def test_executed_submission_matches_full_original_field():
    with np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as fresh,np.load(ROOT/'notebooks/reference_outputs/upstream.npz',allow_pickle=False) as original:
        a,b=fresh['reconstruction'],original['reconstruction']
        assert a.shape==(1,2,200,200) and a.dtype==np.float64
        scale=np.max(np.abs(b));assert np.max(np.abs(a-b))/scale<=2e-8
        np.testing.assert_allclose(a,b,rtol=2e-7,atol=2e-8*scale)

def test_full_physical_velocity_conversion_matches_original():
    data=load_observations()
    with np.load(ROOT/'notebooks/reference_outputs/fresh.npz',allow_pickle=False) as fresh,np.load(ROOT/'notebooks/reference_outputs/upstream.npz',allow_pickle=False) as original:
        velocity=velocity_from_mapping(data['x'],data['y'],fresh['phix'],fresh['phiy'])
        expected=original['velocity'][0]
        scale=np.max(np.abs(expected))
        assert np.max(np.abs(velocity-expected))/scale<=2e-8

def test_fresh_standard_submission_passes_independent_standalone_judge(tmp_path):
    report=tmp_path/'report.json'
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(report.read_text())['passed']
