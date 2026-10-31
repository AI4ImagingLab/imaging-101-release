import importlib.util,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('hdx_judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def data():
 with np.load(ROOT/'output/reconstruction.npz') as a:return {k:a[k].copy() for k in a.files}

def test_fresh_pass():
 r,c=j.evaluate(ROOT/'output/reconstruction.npz');assert c==0 and r['passed'];assert np.isfinite(list(r['metrics'].values())).all()

@pytest.mark.parametrize('case',['missing','key','shape','dtype','complex','nan','inf','huge'])
def test_malformed(tmp_path,case):
 d=data();p=tmp_path/'bad.npz'
 if case=='missing':pass
 elif case=='key':d.pop('uptake')
 elif case=='shape':d['reconstruction']=d['reconstruction'][:,:-1]
 elif case=='dtype':d['uptake']=d['uptake'].astype('float32')
 elif case=='complex':d['uptake']=d['uptake'].astype('complex128')
 elif case in ['nan','inf']:d['uptake'][0,0,0]=float(case)
 else:d['reconstruction'][0,::2]=1e308;d['reconstruction'][0,1::2]=-1e308
 if case!='missing':np.savez(p,**d)
 r,c=j.evaluate(p);assert c==2 and r['metrics']=={} and r['errors'];json.dumps(r,allow_nan=False)

def test_quality_and_every_output(tmp_path):
 d=data();d['reconstruction'][:]=0;d['uptake'][:]=0;p=tmp_path/'poor.npz';np.savez(p,**d);r,c=j.evaluate(p);assert c==1
 d=data();d['uptake'][0,-1,-1]+=1;np.savez(p,**d);r,c=j.evaluate(p);assert c==1 and r['metrics']['forward_observation_normalized_l2']>1e-9

def test_reference_precedence_and_error(tmp_path):
 for sub in ['data','evaluation']:shutil.copytree(ROOT/sub,tmp_path/sub)
 d=data();d['reconstruction'][0,0]+=1000
 from scipy.special import expit
 with np.load(ROOT/'data/raw_data.npz') as a:d['uptake'][0]=j.forward(d['reconstruction'][0],a['membership'][0],a['intrinsic_rates'][0],a['time_s'][0])
 np.savez(tmp_path/'evaluation/ground_truth.npz',**d)
 r,c=j.evaluate(ROOT/'output/reconstruction.npz',tmp_path);assert c==2 and 'truth' in r['errors'][0]
 cfg=json.loads((tmp_path/'evaluation/metrics.json').read_text());cfg['truth_relative_l2_boundary']=1e-12;(tmp_path/'evaluation/metrics.json').write_text(json.dumps(cfg))
 r,c=j.evaluate(ROOT/'output/reconstruction.npz',tmp_path);assert c==1 and r['metrics']['truth_relative_l2_reconstruction']>1e-12
 (tmp_path/'evaluation/ground_truth.npz').unlink();(tmp_path/'evaluation/baseline_reference.npz').unlink();assert j.evaluate(ROOT/'output/reconstruction.npz',tmp_path)[1]==2

def test_foreign_cwd_cli_and_write_failure(tmp_path):
 command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
 assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==0
 (tmp_path/'blocked').write_text('file');command[-1]=str(tmp_path/'blocked/report.json');p=subprocess.run(command,cwd=tmp_path,capture_output=True);assert p.returncode==2 and b'Report write failed' in p.stderr
