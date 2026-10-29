from pathlib import Path
import sys,importlib.util,subprocess,os,json
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def truth():
 with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as d:return d['reconstruction'].copy()
def test_known_metric_case():
 a=np.array([0.,1.,2.]);b=np.array([0.,2.,4.]);m=judge.numerical_metrics(a,b)
 assert np.isclose(m['ncc'],1.) and np.isclose(m['nrmse'],np.sqrt(5/3)/4)
@pytest.mark.parametrize('kind',['zero','scaled','shuffled'])
def test_poor_finite_predictions_fail(tmp_path,kind):
 a=truth()
 if kind=='zero':a[:]=0
 elif kind=='scaled':a*=.5
 else:a=np.roll(a,17,axis=2)
 p=tmp_path/'poor.npz';np.savez_compressed(p,reconstruction=a);r,c=judge.evaluate(p)
 assert c==1 and r['status']=='fail' and not r['passed'] and r['metrics']
@pytest.mark.parametrize('kind',['missing_key','shape','dtype','complex','nan','inf','negative','huge'])
def test_malformed_inputs(tmp_path,kind):
 a=truth();key='reconstruction'
 if kind=='missing_key':key='other'
 elif kind=='shape':a=a[0]
 elif kind=='dtype':a=a.astype(np.float32)
 elif kind=='complex':a=a.astype(np.complex128)
 elif kind=='negative':a.flat[0]=-1
 elif kind=='huge':a.flat[0]=1e308
 elif kind=='nan':a.flat[0]=np.nan
 elif kind=='inf':a.flat[0]=np.inf
 p=tmp_path/'bad.npz';np.savez_compressed(p,**{key:a});r,c=judge.evaluate(p)
 assert c==2 and r['status']=='error' and r['metrics']=={} and r['errors']
 json.dumps(r,allow_nan=False)
def test_missing_and_corrupt_files(tmp_path):
 assert judge.evaluate(tmp_path/'absent.npz')[1]==2
 p=tmp_path/'broken.npz';p.write_bytes(b'not an npz');assert judge.evaluate(p)[1]==2
def test_reference_precedence_missing_assets_and_invalid_metric(tmp_path,monkeypatch):
 monkeypatch.setattr(judge,'HERE',tmp_path)
 settings=(ROOT/'evaluation/metrics.json').read_text(encoding='utf8');(tmp_path/'metrics.json').write_text(settings,encoding='utf8')
 p=tmp_path/'prediction.npz';np.savez_compressed(p,reconstruction=truth())
 assert judge.evaluate(p)[1]==2
 np.savez_compressed(tmp_path/'ground_truth.npz',reconstruction=np.zeros_like(truth()))
 np.savez_compressed(tmp_path/'baseline_reference.npz',reconstruction=truth())
 r,c=judge.evaluate(p);assert c==2 and 'dynamic range' in r['errors'][0]
 (tmp_path/'metrics.json').write_text('{}',encoding='utf8');assert judge.evaluate(p)[1]==2
def test_cli_other_cwd_report_directories_and_write_failure(tmp_path):
 p=tmp_path/'valid.npz';np.savez_compressed(p,reconstruction=truth());r=tmp_path/'nested'/'report.json'
 cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(r)]
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 run=subprocess.run(cmd,cwd=tmp_path,env=env,capture_output=True);assert run.returncode==0 and json.loads(r.read_text())['passed']
 run=subprocess.run(cmd[:-1]+[str(tmp_path)],cwd=tmp_path,env=env,capture_output=True);assert run.returncode==2 and b'Report-write failure' in run.stderr
