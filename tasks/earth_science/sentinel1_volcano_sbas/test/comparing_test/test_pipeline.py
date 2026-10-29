from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('standalone_judge', ROOT/'evaluation/judge.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def test_fresh_cli_output_full_scene_and_all_required_quantities():
    # The required documented CLI precedes pytest; this checks its real saved result.
    prediction = ROOT/'output/reconstruction.npz'
    assert prediction.is_file(), 'Run python -m src.main before the full pipeline verification'
    report, code = judge.evaluate(prediction)
    assert code==0 and report['passed']
    assert len(report['metrics'])==11
    assert report['metrics']['observation_coherence'] >= .98
    with np.load(ROOT/'notebooks/reference_outputs/cleaned_measurements.npz', allow_pickle=False) as measured:
        names = measured['metric_names']
        np.testing.assert_allclose([report['metrics'][str(k)] for k in names], measured['values'], atol=1e-12, rtol=1e-12)


def test_judge_rejects_valid_zero_reconstruction(tmp_path):
    file = tmp_path/'zero.npz'
    np.savez_compressed(file, **{k:np.zeros(shape, np.float32) for k,shape in judge.EXPECTED.items()})
    report, code = judge.evaluate(file)
    assert code==1 and report['status']=='fail' and not report['passed']
    assert report['metrics'] and report['errors']==[]


def test_judge_standalone_paths_and_report_contract_from_other_cwd(tmp_path):
    report_path = tmp_path/'nested/result.json'
    result = subprocess.run([sys.executable, str(ROOT/'evaluation/judge.py'), '--prediction', str(ROOT/'output/reconstruction.npz'), '--report', str(report_path)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode==0, result.stderr
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert set(report)=={'status','passed','metrics','thresholds','errors'}
    assert report['status']=='pass' and report['errors']==[]


def test_cli_help_and_import_preserve_notebook_backend():
    result = subprocess.run([sys.executable, '-m', 'src.main', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode==0 and '--workers' in result.stdout
    import matplotlib
    before = matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before
