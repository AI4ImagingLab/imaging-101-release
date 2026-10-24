from pathlib import Path
import os,json,time,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture
def data():
    from src.preprocessing import load_observations
    import gc
    value=load_observations(ROOT/'data/raw_data.npz')
    yield value
    del value
    gc.collect()

def relative(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel();numerator=0.;denominator=0.
    for start in range(0,a.size,2**18):
        aa=a[start:start+2**18].astype(np.float64);bb=b[start:start+2**18].astype(np.float64)
        numerator+=float(np.sum((aa-bb)**2));denominator+=float(bb@bb)
    return np.sqrt(numerator/denominator)

def test_original_physical_coordinate_view():
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as raw,np.load(ROOT/'test/fixtures/native_ccm_physical_view.npz',allow_pickle=False) as original:
        assert np.array_equal(raw['attitude_coordinates'][0]-1,original['coordinates'])
        assert np.array_equal(raw['attitude_weights_s'][0],original['weights'])

def test_original_full_operators(data):
    with np.load(ROOT/'test/fixtures/native_full_operator_fixtures.npz',allow_pickle=False) as fixture:
        expected=fixture['source_expectation'].reshape(278,-1)
        actual=data.operator.forward(fixture['model'])
        assert relative(actual,expected)<1e-5
        assert relative(data.operator.adjoint(expected),fixture['transposed_source'])<1e-5
    assert relative(data.operator.compute_exposure(),data.operator.exposure)<1e-5

def test_original_first_inverse_update(data):
    from src.solvers import reconstruct
    with np.load(ROOT/'test/fixtures/float32_first_iteration.npz',allow_pickle=False) as fixture:
        result=reconstruct(data,iterations=1)
        assert relative(result.reconstruction,fixture['model'])<1e-5
        assert relative(result.delta,fixture['delta'])<1e-5

def test_standard_full_cli_and_independent_judge(tmp_path):
    # Release the module-scoped data fixture before requesting another complete
    # operator in a subprocess; this test itself does not require that fixture.
    logs=Path(os.environ.get('IMAGING101_TEST_LOG_DIR',str(tmp_path)));logs.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    child=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,shell=False)
    (logs/'pipeline_stdout.log').write_bytes(child.stdout);(logs/'pipeline_stderr.log').write_bytes(child.stderr)
    (logs/'pipeline_execution.json').write_text(json.dumps(dict(returncode=child.returncode,seconds=time.perf_counter()-started)))
    assert child.returncode==0,child.stderr.decode(errors='replace')
    child=subprocess.run([sys.executable,'evaluation/judge.py','--prediction','output/reconstruction.npz','--report','output/judge_report.json'],cwd=ROOT,capture_output=True,shell=False)
    assert child.returncode==0,child.stderr.decode(errors='replace')
    with np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as fresh,np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as source:
        assert relative(fresh['reconstruction'],source['reconstruction'])<1e-5
