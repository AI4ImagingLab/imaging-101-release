from pathlib import Path
import subprocess,sys,json,shutil
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
def invoke(pred,report,judge=None):return subprocess.run([sys.executable,str(judge or T/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],capture_output=True,text=True,encoding='utf8')
def source():
 with np.load(T/'evaluation/baseline_reference.npz') as f:return {k:f[k] for k in f.files}
def test_original_pass_foreigncwd(tmp_path):
 p=invoke(T/'evaluation/baseline_reference.npz',tmp_path/'r.json');assert p.returncode==0,p.stderr
 assert json.loads((tmp_path/'r.json').read_text())['passed']
@pytest.mark.parametrize('kind',['missing','key','shape','dtype','nan','inf'])
def test_malformed(tmp_path,kind):
 a=source();p=tmp_path/'p.npz'
 if kind=='key':del a['reconstruction']
 elif kind=='shape':a['reconstruction']=a['reconstruction'][:,:,:4]
 elif kind=='dtype':a['reconstruction']=a['reconstruction'].astype(np.float32)
 elif kind in ['nan','inf']:a['modeled_spectra'][0,0,0]=float(kind)
 if kind!='missing':np.savez(p,**a)
 r=tmp_path/'report.json';v=invoke(p,r);assert v.returncode==2;z=json.loads(r.read_text());assert z['metrics']=={} and z['errors']
@pytest.mark.parametrize('quantity',[0,1,2,3,4,'model'])
def test_each_output_role_rejects(tmp_path,quantity):
 a=source()
 if quantity=='model':a['modeled_spectra'][:]=0
 else:a['reconstruction'][0,:,quantity]*=4
 p=tmp_path/'p.npz';np.savez(p,**a);r=tmp_path/'r.json';v=invoke(p,r);assert v.returncode==1

def test_assets_and_report_failure(tmp_path):
 copy=tmp_path/'eval';copy.mkdir();shutil.copy(T/'evaluation/judge.py',copy/'judge.py');r=tmp_path/'r.json';assert invoke(T/'evaluation/baseline_reference.npz',r,copy/'judge.py').returncode==2
 blocker=tmp_path/'block';blocker.write_text('file');v=invoke(T/'evaluation/baseline_reference.npz',blocker/'r.json');assert v.returncode==2 and 'report write failed' in v.stderr

@pytest.mark.parametrize('bad',['empty','missing','comparison','nan','negative','zero'])
def test_invalid_evaluator_configuration(tmp_path,bad):
 import copy
 cfg=json.loads((T/'evaluation/metrics.json').read_text());cfg=copy.deepcopy(cfg)
 if bad=='empty':cfg['thresholds']={}
 elif bad=='missing':del cfg['thresholds']['mean_absolute_spectral_error']
 elif bad=='comparison':cfg['thresholds']['mean_absolute_spectral_error']['comparison']='>'
 else:cfg['thresholds']['mean_absolute_spectral_error']['boundary']={'nan':float('nan'),'negative':-1,'zero':0}[bad]
 dest=tmp_path/'evaluation';dest.mkdir();shutil.copy(T/'evaluation/judge.py',dest/'judge.py');(dest/'metrics.json').write_text(json.dumps(cfg));r=tmp_path/'r.json'
 assert invoke(T/'evaluation/baseline_reference.npz',r,dest/'judge.py').returncode==2
 result=json.loads(r.read_text());assert result['metrics']=={} and result['errors']
