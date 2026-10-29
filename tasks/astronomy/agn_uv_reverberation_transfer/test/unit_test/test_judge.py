from pathlib import Path
import importlib.util,shutil,numpy as np,pytest,subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def valid():
 with np.load(ROOT/'evaluation/baseline_reference.npz') as z:return {k:z[k] for k in z.files}
def test_reference_pass():assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz')['passed']
@pytest.mark.parametrize('kind',['missing','shape','dtype','complex','nan','inf','poor','transfer','continuum'])
def test_bad_inputs(tmp_path,kind):
 d=valid()
 if kind=='missing':del d['reconstruction']
 elif kind=='shape':d['reconstruction']=d['reconstruction'][0]
 elif kind=='dtype':d['reconstruction']=d['reconstruction'].astype(np.float16)
 elif kind=='complex':d['reconstruction']=d['reconstruction'].astype(complex)
 elif kind in ('nan','inf'):d['reconstruction'][0,0,0]=getattr(np,kind)
 elif kind=='poor':d['reconstruction']*=0
 elif kind=='transfer':d['transfer'][:,:,0]+=10
 elif kind=='continuum':d['continuum']*=2
 p=tmp_path/'p.npz';np.savez(p,**d);r=judge.evaluate(p);assert r['status']==('fail' if kind in ('poor','transfer','continuum') else 'error')
 if r['status']=='error':assert r['metrics']=={}
def test_missing_file(tmp_path):assert judge.evaluate(tmp_path/'none')['status']=='error'
def test_float32(tmp_path):
 p=tmp_path/'p.npz';np.savez(p,**{k:v.astype('float32') for k,v in valid().items()});assert judge.evaluate(p)['passed']
def test_cli_and_report_failure(tmp_path):
 prediction=ROOT/'evaluation/baseline_reference.npz';report=tmp_path/'new/r.json'
 args=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
 p=subprocess.run(args,cwd=tmp_path,capture_output=True);assert p.returncode==0 and json.loads(report.read_text())['passed']
 args[-1]=str(tmp_path);p=subprocess.run(args,cwd=tmp_path,capture_output=True);assert p.returncode==2 and b'Report write failed' in p.stderr
def test_missing_reference_and_truth_priority(tmp_path,monkeypatch):
 folder=tmp_path/'evaluation';folder.mkdir();shutil.copy(ROOT/'evaluation/metrics.json',folder/'metrics.json');monkeypatch.setattr(judge,'HERE',folder)
 assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz')['status']=='error'
 shutil.copytree(ROOT/'data',tmp_path/'data');d=valid();shutil.copy(ROOT/'evaluation/baseline_reference.npz',folder/'baseline_reference.npz');d['reconstruction']+=1;np.savez(folder/'ground_truth.npz',**d)
 assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz')['status']=='fail'
def test_known_metric_case(tmp_path):
 d=valid();d['reconstruction']+=.002;p=tmp_path/'p.npz';np.savez(p,**d);r=judge.evaluate(p)
 assert abs(r['metrics']['profile_rmse_0']-.002)<1e-12
