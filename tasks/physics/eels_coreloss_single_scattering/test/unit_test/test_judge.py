from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_eels_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def ref():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:return f['reconstruction'].copy()
@pytest.mark.parametrize('kind',['missing_key','shape','integer','complex','nan','inf','missing','garbage'])
def test_malformed(kind,tmp_path):
    p=tmp_path/'p.npz';a=ref()
    if kind=='missing':pass
    elif kind=='garbage':p.write_bytes(b'not npz')
    elif kind=='missing_key':np.savez(p,wrong=a)
    else:
        if kind=='shape':a=a[0]
        if kind=='integer':a=a.astype(np.int64)
        if kind=='complex':a=a.astype(np.complex128)
        if kind in ['nan','inf']:a[0,0]=float(kind)
        np.savez(p,reconstruction=a)
    report,code=judge.evaluate(p)
    assert code==2 and not report['passed'] and report['status']=='error' and report['metrics']=={} and report['errors']
@pytest.mark.parametrize('factor',[0,.2,5,-1])
def test_bad_intensity(factor,tmp_path):
    p=tmp_path/'p.npz';np.savez(p,reconstruction=ref()*factor)
    report,code=judge.evaluate(p);assert code==1 and report['status']=='fail'
def test_plural_and_reversed_negative(tmp_path):
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as f:plural=f['edge'][None]/.2
    for a in [plural,ref()[:,::-1]]:
        p=tmp_path/'p.npz';np.savez(p,reconstruction=a);assert judge.evaluate(p)[1]==1
def test_float32_and_known_metric(tmp_path):
    p=tmp_path/'p.npz';a=ref();np.savez(p,reconstruction=a.astype(np.float32));assert judge.evaluate(p)[1]==0
    np.savez(p,reconstruction=a+np.ptp(a)*.1);r,code=judge.evaluate(p)
    assert code==1 and abs(r['metrics']['nrmse']-.1)<1e-12
def test_evaluator_assets_and_write_error(tmp_path):
    p=tmp_path/'p.npz';np.savez(p,reconstruction=ref())
    r,code=judge.evaluate(p,asset_root=tmp_path/'absent');assert code==2 and r['metrics']=={} and 'Evaluator assets' in r['errors'][0]
    blocker=tmp_path/'blocker';blocker.write_text('file')
    proc=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(blocker/'report.json')],capture_output=True,text=True,encoding='utf-8')
    assert proc.returncode==2 and 'Cannot write report' in proc.stderr
