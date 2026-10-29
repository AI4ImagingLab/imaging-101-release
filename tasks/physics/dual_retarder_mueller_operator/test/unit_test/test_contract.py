from pathlib import Path
import importlib.util,subprocess,sys,json,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_mueller_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def truth():
    with np.load(ROOT/'evaluation/ground_truth.npz',allow_pickle=False) as z:return z['reconstruction'].copy()

def test_independent_known_metrics_and_scale(truth,tmp_path):
    with np.load(ROOT/'data/raw_data.npz') as z:angles=z['angles'][0];power=z['power'][0]
    m=judge.score_arrays(truth,truth,angles,power)
    assert m['ncc_vs_truth']==pytest.approx(1) and m['nrmse_vs_truth']==0
    p=tmp_path/'p.npz';np.savez_compressed(p,reconstruction=truth*2)
    report,code=judge.evaluate(p);assert code==1
    assert report['metrics']['ncc_vs_truth']==pytest.approx(1)
    expected=np.sqrt(np.mean(truth**2))/np.ptp(truth)
    assert report['metrics']['nrmse_vs_truth']==pytest.approx(expected)
    assert report['metrics']['observation_relative_l2']==pytest.approx(1)
    np.savez_compressed(p,reconstruction=np.zeros_like(truth));report,code=judge.evaluate(p)
    assert code==1 and report['metrics']['ncc_vs_truth']==0

def test_all16_elements_are_scored(truth,tmp_path):
    p=tmp_path/'p.npz'
    for row in range(4):
        for col in range(4):
            matrix=truth.copy();matrix[0,row,col]+=.001
            np.savez_compressed(p,reconstruction=matrix)
            report,code=judge.evaluate(p);assert code==1,(row,col,report)

@pytest.mark.parametrize('kind',['missing_key','shape','float32','complex','object','nan','inf'])
def test_malformed_input(truth,tmp_path,kind):
    if kind=='shape':truth=truth[0]
    elif kind=='float32':truth=truth.astype('float32')
    elif kind=='complex':truth=truth.astype('complex128')
    elif kind=='object':truth=truth.astype(object)
    elif kind in ['nan','inf']:truth[0,0,0]=np.nan if kind=='nan' else np.inf
    p=tmp_path/'p.npz';np.savez_compressed(p,**{'wrong' if kind=='missing_key' else 'reconstruction':truth})
    report,code=judge.evaluate(p);assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_missing_and_evaluator_errors(truth,tmp_path):
    assert judge.evaluate(tmp_path/'missing.npz')[1]==2
    p=tmp_path/'p.npz';np.savez_compressed(p,reconstruction=truth)
    assert judge.evaluate(p,asset_dir=tmp_path)[1]==2
    assert judge.evaluate(p,data_path=tmp_path/'missing_raw.npz')[1]==2

def test_true_reference_priority(truth,tmp_path):
    for name in ['metrics.json','ground_truth.npz']:shutil.copy2(ROOT/'evaluation'/name,tmp_path/name)
    np.savez_compressed(tmp_path/'baseline_reference.npz',reconstruction=np.zeros_like(truth))
    np.savez_compressed(tmp_path/'p.npz',reconstruction=truth)
    assert judge.evaluate(tmp_path/'p.npz',asset_dir=tmp_path,data_path=ROOT/'data/raw_data.npz')[1]==0

def test_caller_paths_and_report_failure(truth,tmp_path):
    np.savez_compressed(tmp_path/'p.npz',reconstruction=truth)
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction','p.npz','--report','nested/report.json']
    r=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True);assert r.returncode==0,r.stderr
    assert json.loads((tmp_path/'nested/report.json').read_text())['passed']
    (tmp_path/'blocked').write_text('file');cmd[-1]='blocked/report.json'
    r=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True);assert r.returncode==2 and 'Cannot write report' in r.stderr

def test_pipeline_and_independent_forward(tmp_path):
    from src.main import run
    from src.physics_model import forward
    p=tmp_path/'fresh.npz';out,d,elapsed=run(ROOT/'data/raw_data.npz',p)
    assert elapsed>0 and judge.evaluate(p)[1]==0
    with np.load(ROOT/'data/raw_data.npz') as z:
        angles=z['angles'][0]
        np.testing.assert_allclose(judge.predict_intensity(out['reconstruction'],angles),forward(out['reconstruction'][0],angles),rtol=1e-13,atol=1e-14)

def test_public_contract_and_no_leakage():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    assert not any(ord(c)<32 and c!='\n' for c in text)
    assert 'author-simulated' in text and '24' in text and '5' in text
    for key in ['angles','power']:assert f'| `{key}` |' in text
    assert not any(x in text.lower() for x in ['github.com','githubusercontent','github.io','codeload.','git.io','20261026'])
