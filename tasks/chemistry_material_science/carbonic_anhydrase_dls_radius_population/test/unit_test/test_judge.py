from pathlib import Path
import sys,importlib.util,json,shutil,subprocess
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def good():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as a:return {k:a[k].copy() for k in a.files}

def test_original_pass_and_objective(tmp_path):
    d=good();p=tmp_path/'p.npz';np.savez(p,**d);r,c=j.evaluate(p);assert c==0
    f=np.load(ROOT/'test/fixtures/upstream.npz');np.testing.assert_allclose([r['metrics']['objective_0'],r['metrics']['objective_1']],f['output_residuals']**2,rtol=1e-10)

def test_zero_and_wrong_prediction_fail(tmp_path):
    for name in ['reconstruction','correlation']:
        d=good();d[name]*=0;p=tmp_path/(name+'.npz');np.savez(p,**d);r,c=j.evaluate(p);assert c==1 and not r['passed']

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','negative','corrupt'])
def test_malformed(tmp_path,kind):
    d=good();p=tmp_path/'p.npz'
    if kind=='key':del d['reconstruction']
    if kind=='shape':d['correlation']=d['correlation'][0]
    if kind=='dtype':d['reconstruction']=d['reconstruction'].astype('float32')
    if kind=='complex':d['reconstruction']=d['reconstruction'].astype('complex128')
    if kind in ['nan','inf']:d['reconstruction'][0,0,0]=float(kind)
    if kind=='negative':d['reconstruction'][0,0,0]=-1
    if kind=='corrupt':p.write_text('bad')
    elif kind!='missing':np.savez(p,**d)
    r,c=j.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('fault',['missing','boundary','observations'])
def test_evaluator_errors(tmp_path,fault):
    if fault!='missing':
        shutil.copytree(ROOT/'evaluation',tmp_path/'evaluation');shutil.copytree(ROOT/'data',tmp_path/'data')
    if fault=='boundary':
        cfg=json.loads((tmp_path/'evaluation/metrics.json').read_text());cfg['objective_boundary']=[-1,1];(tmp_path/'evaluation/metrics.json').write_text(json.dumps(cfg))
    if fault=='observations':(tmp_path/'data/raw_data.npz').write_text('bad')
    r,c=j.evaluate(ROOT/'output/reconstruction.npz',tmp_path);assert c==2 and r['metrics']=={}

def test_cli_paths_and_write_error(tmp_path):
    p=tmp_path/'report/report.json';args=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(p)]
    r=subprocess.run(args,cwd=tmp_path,capture_output=True);assert r.returncode==0 and json.loads(p.read_text())['passed']
    blocker=tmp_path/'file';blocker.write_text('x');args[-1]=str(blocker/'r.json');r=subprocess.run(args,cwd=tmp_path,capture_output=True);assert r.returncode==2 and b'Report write failed' in r.stderr


def test_extreme_finite_prediction_rejected_without_nonfinite_report(tmp_path):
    d=good();d['reconstruction'][:]=1e308;p=tmp_path/'p.npz';np.savez(p,**d)
    with np.errstate(all='ignore'):r,c=j.evaluate(p)
    assert c==2 and r['metrics']=={};json.dumps(r,allow_nan=False)

def test_truth_priority_and_required_truth_configuration(tmp_path):
    shutil.copytree(ROOT/'evaluation',tmp_path/'evaluation');shutil.copytree(ROOT/'data',tmp_path/'data');d=good();np.savez(tmp_path/'evaluation/ground_truth.npz',**d)
    altered=good();altered['reconstruction'][:]=0;np.savez(tmp_path/'evaluation/baseline_reference.npz',**altered)
    p=tmp_path/'p.npz';np.savez(p,**d);r,c=j.evaluate(p,tmp_path);assert c==2 and 'Actual truth requires' in r['errors'][0]
    cfg=json.loads((tmp_path/'evaluation/metrics.json').read_text());cfg['truth_relative_l2_boundary']=1e-10;(tmp_path/'evaluation/metrics.json').write_text(json.dumps(cfg));r,c=j.evaluate(p,tmp_path);assert c==0 and r['metrics']['truth_relative_l2_reconstruction']==0
