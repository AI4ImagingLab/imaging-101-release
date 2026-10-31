from pathlib import Path
import importlib.util,subprocess,sys,json,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('palsjudge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def valid():
 with np.load(ROOT/'output/reconstruction.npz') as a:return {k:a[k].copy() for k in a.files}
def test_fresh_and_independent_forward():
 r,c=j.evaluate(ROOT/'output/reconstruction.npz');assert c==0 and r['passed'];assert all(np.isfinite(v) for v in r['metrics'].values())
@pytest.mark.parametrize('case',['file','key','shape','dtype','complex','nan','inf','negative','huge','order'])
def test_malformed(tmp_path,case):
 d=valid();p=tmp_path/'bad.npz'
 if case=='file':pass
 elif case=='key':d.pop('irf_weight')
 elif case=='shape':d['prediction']=d['prediction'][:,:,:-1]
 elif case=='dtype':d['prediction']=d['prediction'].astype('float32')
 elif case=='complex':d['prediction']=d['prediction'].astype('complex128')
 elif case in ['nan','inf']:d['prediction'][0,0,0]=float(case)
 elif case=='negative':d['irf_fwhm_ps'][0,0,0]=-1
 elif case=='huge':d['irf_fwhm_ps'][0,:,:]=1e308
 else:d['reconstruction']=d['reconstruction'][:,:,::-1]
 if case!='file':np.savez(p,**d)
 r,c=j.evaluate(p);assert c==2 and r['metrics']=={} and r['errors'];json.dumps(r,allow_nan=False)
@pytest.mark.parametrize('key',list(j.SHAPES))
def test_every_output_scored(tmp_path,key):
 d=valid();d[key]*=1.2;p=tmp_path/'changed.npz';np.savez(p,**d);r,c=j.evaluate(p);assert c==1 and not r['passed']
def test_truth_precedence_and_evaluator_failure(tmp_path):
 for sub in ['data','evaluation']:shutil.copytree(ROOT/sub,tmp_path/sub)
 d=valid();d['reconstruction']*=1.001
 np.savez(tmp_path/'evaluation/ground_truth.npz',**d);r,c=j.evaluate(ROOT/'output/reconstruction.npz',tmp_path);assert c==2
 (tmp_path/'evaluation/ground_truth.npz').unlink();(tmp_path/'evaluation/baseline_reference.npz').unlink();assert j.evaluate(ROOT/'output/reconstruction.npz',tmp_path)[1]==2
 # Actual truth is chosen ahead of a deliberately broken baseline.
 d=valid();np.savez(tmp_path/'evaluation/ground_truth.npz',**d);cfg=json.loads((tmp_path/'evaluation/metrics.json').read_text());cfg['truth_relative_l2_boundary']=1e-8;(tmp_path/'evaluation/metrics.json').write_text(json.dumps(cfg));assert j.evaluate(ROOT/'output/reconstruction.npz',tmp_path)[1]==0
 d=valid();d['intensity']*=1.001;p=tmp_path/'other.npz';np.savez(p,**d);assert j.evaluate(p,tmp_path)[1]==1

def test_foreign_cli_and_write_error(tmp_path):
 cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')];assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
 (tmp_path/'block').write_text('file');cmd[-1]=str(tmp_path/'block/report.json');r=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert r.returncode==2 and b'Report write failed' in r.stderr
