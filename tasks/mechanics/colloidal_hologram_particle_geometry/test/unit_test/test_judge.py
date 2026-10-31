import subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def run(p,out):return subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(out)],capture_output=True).returncode
def test_judge_contract(tmp_path):
 good=ROOT/'output/reconstruction.npz';assert run(good,tmp_path/'good.json')==0
 assert run(tmp_path/'missing.npz',tmp_path/'missing.json')==2
 cases=[{},dict(reconstruction=np.zeros((1,2,4))),dict(reconstruction=np.zeros((1,2,5),dtype=complex)),dict(reconstruction=np.full((1,2,5),np.nan))]
 for i,c in enumerate(cases):
  p=tmp_path/f'bad{i}.npz';np.savez(p,**c);assert run(p,tmp_path/f'bad{i}.json')==2
 q=np.array([[[900,300,150,1.8,1.7],[300,730,150,1.8,1.7]]],dtype=float);p=tmp_path/'poor.npz';np.savez(p,reconstruction=q);assert run(p,tmp_path/'poor.json')==1

def test_each_physical_quantity_is_evaluated(tmp_path):
 with np.load(ROOT/'output/reconstruction.npz') as f:base=f['reconstruction']
 targets=[(1000.,380.),(400.,800.),(350.,350.),(1.9,1.9),(1.79,1.79)]
 for j,values in enumerate(targets):
  q=base.copy();q[0,:,j]=values;p=tmp_path/f'quantity{j}.npz';np.savez(p,reconstruction=q)
  assert run(p,tmp_path/f'quantity{j}.json')==1

def test_foreign_cwd_and_report_failure(tmp_path):
 p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'foreign.json')],cwd=tmp_path,capture_output=True)
 assert p.returncode==0
 assert run(ROOT/'output/reconstruction.npz',tmp_path)==2

def test_evaluator_assets_and_nonfinite(tmp_path):
 import shutil,json
 evaluator=tmp_path/'evaluation';evaluator.mkdir();shutil.copy(ROOT/'evaluation/judge.py',evaluator/'judge.py')
 def call():return subprocess.run([sys.executable,str(evaluator/'judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'asset.json')],capture_output=True).returncode
 assert call()==2
 shutil.copy(ROOT/'evaluation/metrics.json',evaluator/'metrics.json');assert call()==2
 for value in [np.inf,-np.inf]:
  q=np.full((1,2,5),value);p=tmp_path/'inf.npz';np.savez(p,reconstruction=q);assert run(p,tmp_path/'inf.json')==2
  report=json.loads((tmp_path/'inf.json').read_text());assert report['metrics']=={} and report['status']=='error'
