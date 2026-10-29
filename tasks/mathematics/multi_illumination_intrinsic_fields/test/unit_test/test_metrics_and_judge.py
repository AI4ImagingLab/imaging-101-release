import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',T/'evaluation/judge.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
def test_independent_metric_known_case():
 y=np.tile(np.array([[1.,2.],[3.,4.]]),(10,10));x=y*2;m=np.ones(y.shape,bool);v=J.aligned_metrics(x,y,m)
 assert abs(v['ncc']-1)<1e-15 and v['nrmse']==0 and v['positive_scale']==.5
 x=np.tile(np.array([[4.,3.],[2.,1.]]),(10,10));v=J.aligned_metrics(x,y,m);scale=20/30
 assert abs(v['ncc']-2/3)<1e-15
 assert abs(v['nrmse']-np.sqrt(np.mean((scale*x-y)**2))/3)<1e-15
def test_source_lmse_known_case():
 y=np.arange(1,401,dtype=float).reshape(20,20);m=np.ones(y.shape,bool)
 assert J.local_error(y,2*y,m)==0
 assert J.local_error(y,np.zeros_like(y),m)==1
def run(pred,report):
 p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=report.parent,capture_output=True,text=True)
 return p,json.loads(report.read_text(encoding='utf8'))
def test_foreign_cwd_good_submission(tmp_path):
 p,r=run(T/'output/reconstruction.npz',tmp_path/'report.json');assert p.returncode==0 and r['status']=='pass'
@pytest.mark.parametrize('bad',['missing','keys','shape','float32','complex','nan','inf','nonpositive'])
def test_invalid_prediction_contract(tmp_path,bad):
 with np.load(T/'output/reconstruction.npz') as z:d={k:z[k].copy() for k in z.files}
 f=tmp_path/'submission.npz'
 if bad=='keys':d.pop('shading')
 if bad=='shape':d['shading']=d['shading'][0]
 if bad=='float32':d['shading']=d['shading'].astype('float32')
 if bad=='complex':d['shading']=d['shading'].astype('complex128')
 if bad in ('nan','inf'):d['shading'][0,0,0]=np.nan if bad=='nan' else np.inf
 if bad=='nonpositive':d['shading'][:]=0
 if bad!='missing':np.savez_compressed(f,**d)
 p,r=run(f,tmp_path/'report.json');assert p.returncode==2 and r['status']=='error' and r['metrics']=={} and r['errors']
def test_physically_valid_poor_factorization_fails(tmp_path):
 with np.load(T/'data/raw_data.npz') as z:m=z['object_mask'];image=z['reference_image']
 f=tmp_path/'poor.npz';np.savez_compressed(f,shading=image*m,reflectance=m.astype(float))
 p,r=run(f,tmp_path/'report.json');assert p.returncode==1 and r['status']=='fail' and r['metrics']['product_relative_l2']==0
@pytest.mark.parametrize('bad',[True,'0.9',float('nan'),float('inf'),-1.])
def test_threshold_numeric_guard(bad):
 with pytest.raises(ValueError):J.numeric(bad,'known')
def test_reference_absent_is_error(tmp_path,monkeypatch):
 (tmp_path/'metrics.json').write_bytes((T/'evaluation/metrics.json').read_bytes());monkeypatch.setattr(J,'HERE',tmp_path)
 with pytest.raises(FileNotFoundError):J.evaluate(T/'output/reconstruction.npz')
def test_report_write_failure(tmp_path):
 f=tmp_path/'is_directory';f.mkdir()
 p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'output/reconstruction.npz'),'--report',str(f)],capture_output=True,text=True)
 assert p.returncode==2 and 'Cannot write report' in p.stderr
