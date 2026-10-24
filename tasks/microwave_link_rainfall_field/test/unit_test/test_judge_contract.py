from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_rain_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def case(tmp_path):
 reference=np.linspace(0,40,38063)[None];np.savez(tmp_path/'baseline_reference.npz',reconstruction=reference)
 config=dict(nrmse_boundary=1e-5,negative_count_boundary=0);(tmp_path/'metrics.json').write_text(json.dumps(config))
 return tmp_path,reference,config

def submit(case,array=None,**keys):
 root,reference,_=case;path=root/'submission.npz';np.savez(path,**(keys or dict(reconstruction=reference if array is None else array)));return judge.evaluate(path,root)

def test_known_metrics_and_perfect_field(case):
 _,y,_=case;assert judge.field_error(y+2,y)==pytest.approx(.05)
 r=submit(case);assert r['status']=='pass' and r['metrics']==dict(nrmse=0,negative_count=0)

@pytest.mark.parametrize('bad',[np.zeros((38063,)),np.zeros((1,38062)),np.zeros((1,38063),dtype='complex128'),np.zeros((1,38063),dtype='int64'),np.zeros((1,38063),dtype='float16'),np.full((1,38063),np.nan),np.full((1,38063),np.inf),np.full((1,38063),1e308)])
def test_bad_arrays_and_metric_overflow(case,bad):
 r=submit(case,bad);assert r['status']=='error' and r['metrics']=={} and r['errors'];json.dumps(r,allow_nan=False)

def test_missing_key_and_missing_prediction(case):
 assert submit(case,wrong=np.zeros(1))['status']=='error'
 root,_,_=case;r=judge.evaluate(root/'absent.npz',root);assert r['status']=='error' and r['metrics']=={}

@pytest.mark.parametrize('value',[True,'0',-1,float('nan'),float('inf')])
def test_bad_numeric_config(case,value):
 root,_,config=case;config['nrmse_boundary']=value;(root/'metrics.json').write_text(json.dumps(config));r=submit(case);assert r['status']=='error' and r['metrics']=={}

@pytest.mark.parametrize('value',[.5,38064,True,'0',-1])
def test_bad_count_config(case,value):
 root,_,config=case;config['negative_count_boundary']=value;(root/'metrics.json').write_text(json.dumps(config));assert submit(case)['status']=='error'

def test_actual_truth_precedence_and_invalid_truth_no_fallback(case):
 root,y,_=case;truth=y+1;np.savez(root/'ground_truth.npz',reconstruction=truth)
 assert submit(case,truth)['status']=='pass' and submit(case,y)['status']=='fail'
 np.savez(root/'ground_truth.npz',reconstruction=np.zeros(3));r=submit(case,y);assert r['status']=='error' and r['metrics']=={}

def test_poor_fields_and_physical_nonnegativity(case):
 _,y,_=case
 for x in [np.zeros_like(y),np.full_like(y,y.mean()),np.random.default_rng(3).permutation(y[0])[None]]:assert submit(case,x)['status']=='fail'
 x=y.copy();x[0,0]=-1e-10;r=submit(case,x);assert r['status']=='fail' and r['metrics']['negative_count']==1

def test_degenerate_or_unavailable_reference(case):
 root,_,_=case;np.savez(root/'ground_truth.npz',reconstruction=np.ones((1,38063)));assert submit(case)['status']=='error'
 (root/'ground_truth.npz').unlink();(root/'baseline_reference.npz').unlink();assert submit(case)['status']=='error'

def test_cli_foreign_cwd_exit_codes_and_report_failure(case,tmp_path):
 root,y,_=case;directory=root/'evaluator';directory.mkdir()
 code=directory/'judge.py';code.write_bytes((ROOT/'evaluation/judge.py').read_bytes())
 (directory/'metrics.json').write_bytes((root/'metrics.json').read_bytes());(directory/'baseline_reference.npz').write_bytes((root/'baseline_reference.npz').read_bytes())
 for array,wanted in [(y,0),(np.zeros_like(y),1),(np.zeros(3),2)]:
  np.savez(root/'prediction.npz',reconstruction=array);p=subprocess.run([sys.executable,str(code),'--prediction',str(root/'prediction.npz'),'--report',str(root/'nested/report.json')],cwd=ROOT.parent,capture_output=True)
  assert p.returncode==wanted;json.loads((root/'nested/report.json').read_text())
 bad=root/'not_a_directory';bad.write_text('file')
 p=subprocess.run([sys.executable,str(code),'--prediction',str(root/'prediction.npz'),'--report',str(bad/'report.json')],capture_output=True);assert p.returncode==2 and b'Report write failure' in p.stderr
