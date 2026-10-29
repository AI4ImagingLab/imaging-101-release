from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_uncentered_cosine_dynamic_range_and_degenerate_cases():
    n,e=judge.frame_metrics(np.array([2.,0.]),np.array([1.,2.]))
    assert n==pytest.approx(1/np.sqrt(5))
    assert e==pytest.approx(np.sqrt(2.5))
    assert judge.frame_metrics(np.zeros(2),np.array([1.,2.]))[0]==0
    with pytest.raises(ValueError):judge.frame_metrics(np.ones(2),np.ones(2))
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(ValueError):judge.frame_metrics(np.full(2,1e308),np.array([1.,2.]))

def test_native_judge_foreign_cwd_and_genuine_quality_failure(tmp_path):
    prediction=ROOT/'evaluation/baseline_reference.npz';report=tmp_path/'nested/r.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    p=subprocess.run(command,cwd=tmp_path,capture_output=True);assert p.returncode==0,p.stderr
    assert json.loads(report.read_text())['passed']
    bad=tmp_path/'bad.npz';np.savez_compressed(bad,reconstruction=np.zeros(judge.SHAPE))
    command[3]=str(bad);p=subprocess.run(command,cwd=tmp_path,capture_output=True)
    r=json.loads(report.read_text());assert p.returncode==1 and r['status']=='fail' and not r['errors']

@pytest.mark.parametrize('case',['missing','key','shape','dtype','complex','nan','inf','not_npz'])
def test_malformed_input_reports_empty_metrics(tmp_path,case):
    file=tmp_path/'input.npz';report=tmp_path/'report.json'
    if case=='not_npz':file.write_bytes(b'not a zip')
    elif case!='missing':
        if case=='key':np.savez(file,wrong=np.ones(1))
        elif case=='shape':np.savez(file,reconstruction=np.ones(1))
        else:
            dtype=np.float32 if case=='dtype' else np.complex128 if case=='complex' else np.float64
            value=np.zeros(judge.SHAPE,dtype=dtype)
            if case in ['nan','inf']:value.flat[0]=float(case)
            np.savez_compressed(file,reconstruction=value)
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(file),'--report',str(report)],cwd=tmp_path,capture_output=True)
    r=json.loads(report.read_text());assert p.returncode==2 and r['metrics']=={} and r['errors']

def test_missing_assets_invalid_threshold_and_report_write_failure(tmp_path):
    evaluation=tmp_path/'evaluation';evaluation.mkdir();shutil.copyfile(ROOT/'evaluation/metrics.json',evaluation/'metrics.json')
    r=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',evaluation);assert r['status']=='error' and not r['metrics']
    config=json.loads((evaluation/'metrics.json').read_text());config['anisotropic_tv_boundaries'][0]=-1;(evaluation/'metrics.json').write_text(json.dumps(config))
    assert 'configuration' in judge.evaluate(ROOT/'evaluation/baseline_reference.npz',evaluation)['errors'][0]
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'missing.npz'),'--report',str(tmp_path)],capture_output=True)
    assert p.returncode==2 and b'Report write failed' in p.stderr
