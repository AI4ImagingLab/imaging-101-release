from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
@pytest.fixture
def reference():
 with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return z['reconstruction']
@pytest.mark.parametrize('case',['missing','key','shape','integer','complex','nan','inf','object'])
def test_malformed(tmp_path,reference,case):
 p=tmp_path/'prediction.npz';a=reference.copy()
 if case=='missing':pass
 elif case=='key':np.savez(p,wrong=a)
 elif case=='shape':np.savez(p,reconstruction=a[0])
 elif case=='integer':np.savez(p,reconstruction=a.astype('int32'))
 elif case=='complex':np.savez(p,reconstruction=a.astype('complex128'))
 elif case=='object':np.savez(p,reconstruction=np.array([object()],dtype=object))
 else:a.flat[0]=np.nan if case=='nan' else np.inf;np.savez(p,reconstruction=a)
 r=judge.evaluate(p);assert r['status']=='error' and r['metrics']=={} and r['errors']
@pytest.mark.parametrize('channel',range(4))
def test_all_quantities_required(tmp_path,reference,channel):
 a=reference.copy();a[:,channel]+=1 if channel<2 else 1e-4;p=tmp_path/'p.npz';np.savez(p,reconstruction=a)
 assert judge.evaluate(p)['status']=='fail'
@pytest.mark.parametrize('case',['truth_priority','baseline_only','neither'])
def test_reference_priority(tmp_path,reference,case):
 (tmp_path/'metrics.json').write_text((T/'evaluation/metrics.json').read_text(),encoding='utf8')
 p=tmp_path/'p.npz';np.savez(p,reconstruction=reference)
 if case=='truth_priority':
  np.savez(tmp_path/'ground_truth.npz',reconstruction=reference);np.savez(tmp_path/'baseline_reference.npz',reconstruction=reference+2)
 elif case=='baseline_only':np.savez(tmp_path/'baseline_reference.npz',reconstruction=reference)
 assert judge.evaluate(p,tmp_path)['status']==('error' if case=='neither' else 'pass')
def test_metric_known_case(tmp_path,reference):
 a=reference.copy();a[:,0]+=.01;p=tmp_path/'p.npz';np.savez(p,reconstruction=a)
 r=judge.evaluate(p);assert abs(r['metrics']['remanent_B_rmse']-.01)<1e-12 and r['passed']
def test_cli_foreign_cwd_and_write_error(tmp_path):
 pred=T/'evaluation/baseline_reference.npz';dest=tmp_path/'nested/report.json'
 r=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(dest)],cwd=tmp_path,capture_output=True)
 assert r.returncode==0 and json.loads(dest.read_text())['passed']
 bad=tmp_path/'not_directory';bad.write_text('blocked')
 r=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(bad/'r.json')],capture_output=True)
 assert r.returncode==2 and b'Cannot write report' in r.stderr
def test_invalid_threshold(tmp_path,reference):
 conf=json.loads((T/'evaluation/metrics.json').read_text());conf['B_rmse_boundary']=float('nan')
 (tmp_path/'metrics.json').write_text(json.dumps(conf));np.savez(tmp_path/'baseline_reference.npz',reconstruction=reference)
 assert judge.evaluate(T/'evaluation/baseline_reference.npz',tmp_path)['status']=='error'

@pytest.mark.parametrize('bad_boundary',[True,False,'0.02'])
def test_reject_nonnumeric_json_threshold(tmp_path,reference,bad_boundary):
 conf=json.loads((T/'evaluation/metrics.json').read_text());conf['B_rmse_boundary']=bad_boundary
 (tmp_path/'metrics.json').write_text(json.dumps(conf));np.savez(tmp_path/'baseline_reference.npz',reconstruction=reference)
 result=judge.evaluate(T/'evaluation/baseline_reference.npz',tmp_path)
 assert result['status']=='error' and result['metrics']=={} and 'JSON number' in result['errors'][0]

