from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_observations
from src.physics_model import harmonic_basis,temporal_basis,field_maps
from src.solvers import damping_matrices,reconstruct
ROOT=Path(__file__).resolve().parents[2]

def expand(banded):
    n=banded.shape[1]; a=np.zeros((n,n)); width=banded.shape[0]-1
    for row in range(width+1):
        offset=width-row
        col=np.arange(offset,n); a[col-offset,col]=banded[row,col]; a[col,col-offset]=banded[row,col]
    return a

def test_preprocessing_and_physics_parity():
    z=np.load(ROOT/'test/fixtures/upstream_stages.npz'); d=load_observations(); ix=z['input_indices']
    assert_allclose(d[2][ix],z['output_values'],rtol=0,atol=1e-14)
    assert_allclose(d[3][ix],z['output_sigma'],rtol=0,atol=1e-14)
    assert_allclose(harmonic_basis(d[1])[ix],z['output_spatial'],rtol=1e-11,atol=1e-12)
    assert_allclose(temporal_basis(d[0])[ix],z['output_temporal'],rtol=0,atol=0)

def test_regularizer_parity():
    z=np.load(ROOT/'test/fixtures/upstream_stages.npz'); s,t=damping_matrices()
    assert_allclose(s,expand(z['output_sdamp']),rtol=2e-13,atol=1e-10)
    assert_allclose(t,expand(z['output_tdamp']),rtol=2e-13,atol=1e-13)

def test_updates_and_complete_pipeline():
    z=np.load(ROOT/'test/fixtures/upstream_stages.npz'); c,d=reconstruct(load_observations())
    assert_allclose(d['iterations'],z['output_iterations'],rtol=1e-9,atol=2e-7)
    assert_allclose(c,z['output_solution'],rtol=1e-9,atol=2e-7)
    ref=np.load(ROOT/'evaluation/baseline_reference.npz')
    assert_allclose(field_maps(c)[None],ref['reconstruction'],rtol=1e-9,atol=2e-7)

def test_cli_judge_pipeline(tmp_path):
    import subprocess,sys
    from src.main import main
    path=main()
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(tmp_path/'report.json')],cwd=tmp_path)
    assert result.returncode==0
