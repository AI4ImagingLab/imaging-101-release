from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def valid():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}

def test_independent_known_metrics(tmp_path,valid):
    p=tmp_path/'p.npz';np.savez_compressed(p,**valid)
    report,code=judge.evaluate(p)
    assert code==0 and report['passed']
    valid['reconstruction']*=2
    np.savez_compressed(p,**valid);report,code=judge.evaluate(p)
    assert code==1 and report['status']=='fail'
    assert report['metrics']['eedf_relative_l2']==pytest.approx(1)
    assert report['metrics']['normalization_error']==pytest.approx(1)

@pytest.mark.parametrize('kind',['missing_key','wrong_shape','float32','complex','nan','inf','object'])
def test_malformed(tmp_path,valid,kind):
    if kind=='missing_key':valid.pop('T_e')
    elif kind=='wrong_shape':valid['reconstruction']=valid['reconstruction'][0]
    elif kind=='float32':valid['reconstruction']=valid['reconstruction'].astype('float32')
    elif kind=='complex':valid['reconstruction']=valid['reconstruction'].astype('complex128')
    elif kind=='object':valid['reconstruction']=valid['reconstruction'].astype(object)
    else:valid['T_e'][0,0]=np.nan if kind=='nan' else np.inf
    p=tmp_path/'p.npz';np.savez_compressed(p,**valid)
    report,code=judge.evaluate(p)
    assert code==2 and report['metrics']=={} and report['errors']

def test_every_quantity_is_scored(tmp_path,valid):
    for key in valid:
        changed={k:v.copy() for k,v in valid.items()};changed[key]*=1.01
        p=tmp_path/f'{key}.npz';np.savez_compressed(p,**changed)
        report,code=judge.evaluate(p)
        assert code==1,(key,report)

def test_missing_prediction_and_hidden_assets(tmp_path):
    report,code=judge.evaluate(tmp_path/'absent.npz');assert code==2 and report['metrics']=={}
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path);assert code==2 and report['metrics']=={}

def test_cli_caller_paths_and_report_write_failure(tmp_path,valid):
    np.savez_compressed(tmp_path/'prediction.npz',**valid)
    args=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction','prediction.npz','--report','nested/report.json']
    result=subprocess.run(args,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads((tmp_path/'nested/report.json').read_text())['passed']
    (tmp_path/'blocked').write_text('file')
    args[-1]='blocked/report.json';result=subprocess.run(args,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2 and 'Cannot write report' in result.stderr

def test_local_main_fresh_pipeline(tmp_path):
    from src.main import run
    p=tmp_path/'fresh.npz';out,diag,elapsed=run(ROOT/'data/raw_data.npz',p)
    assert elapsed>0 and p.is_file()
    assert judge.evaluate(p)[1]==0

def test_document_math_and_public_interface():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    assert not any(ord(c)<32 and c!='\n' for c in text)
    for key in ['bias','current']:assert f'| `{key}` |' in text
    assert '1.145' in text and '1.57' in text and '304' in text
    assert '1.5' in text and '0.2' in text and 'edge_order=1' in text
    assert 'first sample' in text and 'saturation proxy' in text
    assert not any(x in text.lower() for x in ['github.com','github.io','githubusercontent','codeload.','git.io'])
