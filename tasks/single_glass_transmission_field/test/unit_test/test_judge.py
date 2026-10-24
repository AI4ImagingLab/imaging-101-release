from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',T/'evaluation/judge.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
def test_independent_known_metrics():
 x=np.array([1.,2.,3.,4.]);y=x[::-1];v=J.reference_metrics(x,y)
 assert abs(v['ncc']-2/3)<1e-15
 assert abs(v['nrmse']-np.sqrt(5)/3)<1e-15
 assert v['maximum_absolute']==3
 assert J.reference_metrics(np.zeros(4),x)['ncc']==0
def test_independent_stationarity_and_flux():
 y=np.full((8,11,3),.37);v=J.physical_metrics(y,y)
 assert v['stationarity_relative']==0 and v['channel_mean_absolute']==0
 rng=np.random.default_rng(93);a=rng.normal(size=(8,11,3));b=rng.normal(size=a.shape)
 np.testing.assert_allclose(np.sum(a*J.laplace(b)),np.sum(b*J.laplace(a)),atol=1e-12)
def invoke(pred,report):
 p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=report.parent,capture_output=True,text=True)
 return p,json.loads(report.read_text(encoding='utf8'))
def test_foreign_cwd_good_submission(tmp_path):
 p,r=invoke(T/'output/reconstruction.npz',tmp_path/'report.json');assert p.returncode==0 and r['passed'] and r['status']=='pass'
@pytest.mark.parametrize('bad',['missing','key','shape','float32','integer','complex','nan','inf'])
def test_invalid_submissions(tmp_path,bad):
 with np.load(T/'output/reconstruction.npz') as z:x=z['reconstruction'].copy()
 key='reconstruction';f=tmp_path/'submission.npz'
 if bad=='key':key='other'
 if bad=='shape':x=x[0]
 if bad=='float32':x=x.astype('float32')
 if bad=='integer':x=x.astype('int32')
 if bad=='complex':x=x.astype('complex128')
 if bad in ('nan','inf'):x.flat[0]=np.nan if bad=='nan' else np.inf
 if bad!='missing':np.savez_compressed(f,**{key:x})
 p,r=invoke(f,tmp_path/'report.json');assert p.returncode==2 and r['status']=='error' and r['metrics']=={} and r['errors']
def test_zero_is_valid_but_fails(tmp_path):
 f=tmp_path/'zero.npz';np.savez_compressed(f,reconstruction=np.zeros(J.SHAPE))
 p,r=invoke(f,tmp_path/'report.json');assert p.returncode==1 and r['status']=='fail'
def test_existing_invalid_truth_never_falls_back(tmp_path):
 (tmp_path/'metrics.json').write_bytes((T/'evaluation/metrics.json').read_bytes())
 (tmp_path/'baseline_reference.npz').write_bytes((T/'evaluation/baseline_reference.npz').read_bytes())
 np.savez_compressed(tmp_path/'ground_truth.npz',reconstruction=np.zeros((1,2,2,3)))
 with pytest.raises(ValueError,match='expected real float64'):J.evaluate(T/'output/reconstruction.npz',tmp_path)
@pytest.mark.parametrize('bad',[True,'0.9',float('nan'),float('inf'),-1.])
def test_invalid_thresholds(tmp_path,bad):
 config=json.loads((T/'evaluation/metrics.json').read_text());config['nrmse_boundary']=bad
 (tmp_path/'metrics.json').write_text(json.dumps(config),encoding='utf8')
 with pytest.raises(ValueError,match='threshold'):J.evaluate(T/'output/reconstruction.npz',tmp_path)
def test_missing_evaluator_assets(tmp_path):
 with pytest.raises(FileNotFoundError):J.evaluate(T/'output/reconstruction.npz',tmp_path)
def test_report_write_failure(tmp_path):
 folder=tmp_path/'directory';folder.mkdir()
 p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'output/reconstruction.npz'),'--report',str(folder)],capture_output=True,text=True)
 assert p.returncode==2 and 'Cannot write report' in p.stderr
