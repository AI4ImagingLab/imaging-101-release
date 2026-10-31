import numpy as np,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def run(p,out,judge=None,cwd=None):return subprocess.run([sys.executable,str(judge or ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(out)],cwd=cwd,capture_output=True).returncode
def test_judge_contract_and_each_output(tmp_path):
 good=ROOT/'output/reconstruction.npz';assert run(good,tmp_path/'good.json',cwd=tmp_path)==0
 assert run(tmp_path/'missing.npz',tmp_path/'missing.json')==2
 with np.load(good) as f:base={k:f[k] for k in f.files}
 for kind in ['missing','shape','complex','nan','inf','integer']:
  d={k:v.copy() for k,v in base.items()}
  if kind=='missing':del d['parameters']
  if kind=='shape':d['reconstruction']=d['reconstruction'][:,:,:3]
  if kind in ['complex','integer']:d['parameters']=d['parameters'].astype(complex if kind=='complex' else int)
  if kind in ['nan','inf']:d['isotropic_stress'][0,3]=np.nan if kind=='nan' else np.inf
  p=tmp_path/f'{kind}.npz';np.savez(p,**d);assert run(p,tmp_path/f'{kind}.json')==2
 for key in base:
  d={k:v.copy() for k,v in base.items()};d[key]+=10.;p=tmp_path/f'poor{key}.npz';np.savez(p,**d);assert run(p,tmp_path/f'poor{key}.json')==1
 with np.load(ROOT/'evaluation/baseline_reference.npz') as f:up={k:f[k] for k in f.files}
 for j in range(12):
  d={k:v.copy() for k,v in up.items()};d['parameters'][0,j]*=1.2;p=tmp_path/f'param{j}.npz';np.savez(p,**d);assert run(p,tmp_path/f'param{j}.json')==1
 assert run(good,tmp_path)==2
 evaluator=tmp_path/'evaluation';evaluator.mkdir();shutil.copy(ROOT/'evaluation/judge.py',evaluator/'judge.py');assert run(good,tmp_path/'asset.json',judge=evaluator/'judge.py')==2
 shutil.copy(ROOT/'evaluation/metrics.json',evaluator/'metrics.json');assert run(good,tmp_path/'asset.json',judge=evaluator/'judge.py')==2


def test_evaluator_configuration_rejected(tmp_path):
 import json,copy
 evaluator=tmp_path/'evaluation';evaluator.mkdir();shutil.copy(ROOT/'evaluation/judge.py',evaluator/'judge.py')
 original=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
 variants=[]
 variants.append({});variants.append(dict(thresholds={}))
 for kind in ['missing_metric','comparison','nan','negative','zero','string']:
  cfg=copy.deepcopy(original)
  if kind=='missing_metric':del cfg['thresholds']['stress_mse']
  elif kind=='comparison':cfg['thresholds']['stress_mse']['comparison']='arbitrary'
  else:cfg['thresholds']['stress_mse']['boundary']={'nan':float('nan'),'negative':-1.,'zero':0.,'string':'0.1'}[kind]
  variants.append(cfg)
 for i,cfg in enumerate(variants):
  (evaluator/'metrics.json').write_text(json.dumps(cfg),encoding='utf8');report=tmp_path/f'config{i}.json'
  assert run(ROOT/'output/reconstruction.npz',report,judge=evaluator/'judge.py')==2
  result=json.loads(report.read_text(encoding='utf8'));assert result['status']=='error' and result['metrics']=={} and result['errors']
