import json,subprocess,sys
import numpy as np
import pytest
def arrays(root):
 with np.load(root/'evaluation/baseline_reference.npz') as z:return {k:z[k].copy() for k in z.files}
def write(tmp_path,a):
 p=tmp_path/'prediction.npz';np.savez_compressed(p,**a);return p
@pytest.mark.parametrize('key',['reconstruction','thermodynamic_parameters','baseline_coefficients'])
def test_every_output_required(root,tmp_path,judge,key):
 a=arrays(root);a[key]+= .1;report,code=judge.evaluate(write(tmp_path,a));assert code==1,report
@pytest.mark.parametrize('defect',['missing','shape','dtype','complex','nan','inf','negative','kelvin'])
def test_invalid(root,tmp_path,judge,defect):
 a=arrays(root)
 if defect=='missing':a.pop('baseline_coefficients')
 elif defect=='shape':a['reconstruction']=a['reconstruction'][0]
 elif defect=='dtype':a['reconstruction']=a['reconstruction'].astype('float32')
 elif defect=='complex':a['reconstruction']=a['reconstruction'].astype('complex128')
 elif defect in ['nan','inf']:a['reconstruction'][0,0,0]=np.nan if defect=='nan' else np.inf
 elif defect=='negative':a['thermodynamic_parameters'][0,1]=-1
 else:a['thermodynamic_parameters'][0,0]=-274.
 r,c=judge.evaluate(write(tmp_path,a));assert c==2 and r['metrics']=={} and r['errors']
def test_missing_assets(tmp_path,judge):
 assert judge.evaluate(tmp_path/'missing.npz')[1]==2;assert judge.evaluate(tmp_path/'missing.npz',evaluation_dir=tmp_path)[1]==2
def test_cli_foreign_cwd_and_write_error(root,tmp_path):
 report=tmp_path/'nested/report.json';args=[sys.executable,str(root/'evaluation/judge.py'),'--prediction',str(root/'evaluation/baseline_reference.npz'),'--report',str(report)];r=subprocess.run(args,cwd=tmp_path,capture_output=True);assert r.returncode==0 and json.loads(report.read_text())['passed']
 obstacle=tmp_path/'file';obstacle.write_text('x');args[-1]=str(obstacle/'report.json');r=subprocess.run(args,cwd=tmp_path,capture_output=True);assert r.returncode==2 and b'Report-write failure' in r.stderr
