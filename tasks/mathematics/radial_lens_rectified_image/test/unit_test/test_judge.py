from pathlib import Path
import sys,importlib.util,shutil,subprocess,json
import numpy as np,pytest
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('lens_judge',T/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def valid():
    with np.load(T/'evaluation/baseline_reference.npz') as f:return {k:f[k] for k in f.files}
@pytest.mark.parametrize('kind',['missing','key','shape','image_dtype','parameter_dtype','nan','inf','complex'])
def test_invalid_contract(tmp_path,kind):
    p=tmp_path/'x.npz';d=valid()
    if kind=='key':del d['distortion']
    if kind=='shape':d['reconstruction']=d['reconstruction'][0]
    if kind=='image_dtype':d['reconstruction']=d['reconstruction'].astype(float)
    if kind=='parameter_dtype':d['distortion']=d['distortion'].astype(np.float32)
    if kind=='nan':d['distortion'][0,2]=np.nan
    if kind=='inf':d['distortion'][0,2]=np.inf
    if kind=='complex':d['distortion']=d['distortion'].astype(complex)
    if kind!='missing':np.savez(p,**d)
    report,code=j.evaluate(p);assert code==2 and not report['metrics'] and report['errors']
@pytest.mark.parametrize('kind',['black','wrong_geometry','gauge','odd'])
def test_valid_poor_submission_fails(tmp_path,kind):
    d=valid()
    if kind=='black':d['reconstruction'][:]=0
    if kind=='wrong_geometry':d['distortion'][0,2]*=2
    if kind=='gauge':d['distortion'][0,0]=2
    if kind=='odd':d['distortion'][0,1]=.01
    p=tmp_path/'x.npz';np.savez(p,**d)
    report,code=j.evaluate(p);assert code==1 and report['status']=='fail'
def test_truth_precedence_and_bad_hidden_assets(tmp_path):
    for n in ['metrics.json','baseline_reference.npz']:shutil.copy(T/'evaluation'/n,tmp_path/n)
    p=tmp_path/'prediction.npz';np.savez(p,**valid());assert j.evaluate(p,tmp_path)[1]==0
    (tmp_path/'ground_truth.npz').write_bytes(b'broken');assert j.evaluate(p,tmp_path)[1]==2
    d=valid();d['distortion'][0,0]=2;np.savez(tmp_path/'ground_truth.npz',**d)
    assert j.evaluate(p,tmp_path)[1]==2
def test_known_metrics():
    a=np.array([[1.,0.,1e-6,0.,1e-11]]);x=np.array([0,1,2],dtype=np.uint8)
    m=j.score(x,a,x,a);assert m['ncc']==pytest.approx(1) and m['nrmse']==0 and m['coefficient_relative']==0

def test_cli_cwd_report_parent_and_write_failure(tmp_path):
    prediction=T/'evaluation/baseline_reference.npz';report=tmp_path/'nested/result.json'
    command=[sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    p=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert p.returncode==0 and json.loads(report.read_text())['passed']
    command[-1]=str(tmp_path)
    p=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert p.returncode==2 and b'Report write failed' in p.stderr

def test_missing_evaluator_assets(tmp_path):
    shutil.copy(T/'evaluation/metrics.json',tmp_path/'metrics.json')
    r,code=j.evaluate(T/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and not r['metrics'] and r['errors']
