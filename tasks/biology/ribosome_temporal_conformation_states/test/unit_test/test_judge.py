import json,subprocess,sys
import numpy as np
import pytest
def arrays(root):
 with np.load(root/'evaluation/baseline_reference.npz') as z:return {k:z[k].copy() for k in z.files}
def write(tmp_path,a):
 p=tmp_path/'prediction.npz';np.savez_compressed(p,**a);return p
def test_global_permutation(root,tmp_path,judge):
 a=arrays(root);valid=a['reconstruction']>=0;a['reconstruction'][valid]=1-a['reconstruction'][valid]
 for k in ['state_mean','state_variance','initial_probability','population_fraction']:a[k]=a[k][:,::-1]
 a['transition_probability']=a['transition_probability'][:,::-1,::-1];assert judge.evaluate(write(tmp_path,a))[1]==0
@pytest.mark.parametrize('key',['state_mean','state_variance','transition_probability','initial_probability','population_fraction','reconstruction'])
def test_every_quantity_required(root,tmp_path,judge,key):
 a=arrays(root)
 if key=='reconstruction':
  valid=a[key]>=0;a[key][valid]=1-a[key][valid]
 elif key=='transition_probability':a[key][0]=[[.5,.5],[.5,.5]]
 elif key in ['initial_probability','population_fraction']:a[key][0]=[.5,.5]
 else:a[key]+= .01
 report,code=judge.evaluate(write(tmp_path,a));assert code==1,report
@pytest.mark.parametrize('defect',['missing','shape','dtype','nan','inf','sentinel','variance','probability'])
def test_malformed(root,tmp_path,judge,defect):
 a=arrays(root)
 if defect=='missing':a.pop('state_mean')
 elif defect=='shape':a['state_mean']=a['state_mean'][0]
 elif defect=='dtype':a['state_mean']=a['state_mean'].astype('float32')
 elif defect in ['nan','inf']:a['state_mean'][0,0]=np.nan if defect=='nan' else np.inf
 elif defect=='sentinel':a['reconstruction'][a['reconstruction']<0]=0
 elif defect=='variance':a['state_variance'][0,0]=0
 else:a['initial_probability'][0]=[.8,.8]
 report,code=judge.evaluate(write(tmp_path,a));assert code==2 and report['metrics']=={} and report['errors']
def test_missing_input_and_evaluator(tmp_path,judge):
 assert judge.evaluate(tmp_path/'absent.npz')[1]==2;assert judge.evaluate(tmp_path/'absent.npz',evaluation_dir=tmp_path)[1]==2
def test_cli_foreign_cwd_and_report_failure(root,tmp_path):
 p=root/'evaluation/baseline_reference.npz';report=tmp_path/'nested/report.json';args=[sys.executable,str(root/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)];r=subprocess.run(args,cwd=tmp_path,capture_output=True);assert r.returncode==0 and json.loads(report.read_text())['passed']
 obstacle=tmp_path/'file';obstacle.write_text('x');args[-1]=str(obstacle/'report.json');r=subprocess.run(args,cwd=tmp_path,capture_output=True);assert r.returncode==2 and b'Report-write failure' in r.stderr
