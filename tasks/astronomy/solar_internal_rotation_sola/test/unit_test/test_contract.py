from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_solar_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def valid():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return z['reconstruction'].copy()

def test_known_metrics_and_poor_outputs(valid,tmp_path):
    p=tmp_path/'p.npz';np.savez_compressed(p,reconstruction=valid)
    report,code=judge.evaluate(p);assert code==0,report
    assert report['metrics']['rotation_max_abs_nHz']==0
    assert report['metrics']['observation_weighted_rms']==pytest.approx(1.1936132278341605,abs=1e-10)
    for x in [np.zeros_like(valid),-valid,valid+10,np.full_like(valid,430.)]:
        np.savez_compressed(p,reconstruction=x);report,code=judge.evaluate(p);assert code==1,report

@pytest.mark.parametrize('kind',['key','shape','float32','complex','nan','inf','object'])
def test_malformed(valid,tmp_path,kind):
    if kind=='shape':valid=valid[0]
    elif kind=='float32':valid=valid.astype('float32')
    elif kind=='complex':valid=valid.astype('complex128')
    elif kind=='object':valid=valid.astype(object)
    elif kind in ['nan','inf']:valid[0,0,0]=np.nan if kind=='nan' else np.inf
    p=tmp_path/'p.npz';np.savez_compressed(p,**{'wrong' if kind=='key' else 'reconstruction':valid})
    report,code=judge.evaluate(p);assert code==2 and report['metrics']=={} and report['errors']

def test_missing_and_evaluator_errors(valid,tmp_path):
    report,code=judge.evaluate(tmp_path/'absent.npz');assert code==2 and report['metrics']=={}
    p=tmp_path/'valid.npz';np.savez_compressed(p,reconstruction=valid)
    assert judge.evaluate(p,asset_dir=tmp_path)[1]==2
    assert judge.evaluate(p,data_path=tmp_path/'absent_calibration.npz')[1]==2

def test_caller_paths_and_report_failure(valid,tmp_path):
    np.savez_compressed(tmp_path/'pred.npz',reconstruction=valid)
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction','pred.npz','--report','nested/report.json']
    result=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True,encoding='utf-8');assert result.returncode==0,result.stderr
    assert json.loads((tmp_path/'nested/report.json').read_text())['passed']
    (tmp_path/'blocked').write_text('file');cmd[-1]='blocked/report.json'
    result=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True,encoding='utf-8');assert result.returncode==2 and 'Cannot write report' in result.stderr

def test_independent_judge_direct_physics():
    from src.preprocessing import load_data
    from src.physics_model import RotationOperator
    d=load_data(ROOT/'data/raw_data.npz');op=RotationOperator(d)
    field=np.full((1,101,51),430.,dtype=np.float64)
    expected=np.sqrt(np.mean((op.forward(field)-d['splittings']*op.inv_error)**2))
    assert judge.observation_rms(field,ROOT/'data/raw_data.npz')==pytest.approx(expected,rel=1e-12)

def test_public_contract_metadata_and_dependency_roles():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        for key in z.files:
            assert f'| `{key}` |' in text
            assert z[key].shape[0]==1
            assert not z[key].dtype.hasobject
    assert '6.67232e-8' in text and '101' in text and '51' in text
    assert not any(s in text.lower() for s in ['github.com','githubusercontent','github.io','codeload.','git.io'])
    for p in [ROOT/'README.md',*sorted((ROOT/'plan').glob('*.md'))]:
        assert not any(ord(c)<32 and c!='\n' for c in p.read_text(encoding='utf8'))
