from pathlib import Path
import json,subprocess,sys,shutil
import numpy as np
import pytest
from evaluation.judge import evaluate,load_arrays,observation_rmse

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('kind',['missing','key','shape','integer','complex','nan','inf','object','outside'])
def test_bad_submission(kind,tmp_path):
    p=tmp_path/'bad.npz';a=load_arrays(ROOT/'evaluation/baseline_reference.npz')
    if kind=='key':del a['center']
    elif kind=='shape':a['width']=a['width'][:,:,:-1]
    elif kind in ('integer','complex','object'):a['width']=a['width'].astype({'integer':int,'complex':complex,'object':object}[kind])
    elif kind in ('nan','inf'):a['width'].flat[0]=float(kind)
    elif kind=='outside':a['amplitude'][0,0,0]=.1
    if kind!='missing':np.savez_compressed(p,**a)
    report,code=evaluate(p)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_valid_poor_field_and_known_rmse(tmp_path):
    a=load_arrays(ROOT/'evaluation/baseline_reference.npz')
    with np.load(ROOT/'data/raw_data.npz') as z:mask=z['observation_mask'][0]
    a['field_shift'][0][mask]+=.01;p=tmp_path/'poor.npz';np.savez_compressed(p,**a)
    report,code=evaluate(p);assert code==1 and not report['passed']
    assert report['metrics']['maximum_voxel_mse_excess']>1e-6

def test_reference_pass_foreign_cwd_report_write_failure(tmp_path):
    p=ROOT/'evaluation/baseline_reference.npz';report=tmp_path/'new/report.json'
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert result.returncode==0 and json.loads(report.read_text())['status']=='pass'
    blocked=tmp_path/'file';blocked.write_text('not a directory')
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(blocked/'bad.json')],capture_output=True)
    assert result.returncode==2 and b'report write failed' in result.stderr

def test_missing_assets_and_invalid_boundary(tmp_path):
    p=ROOT/'evaluation/baseline_reference.npz';report,code=evaluate(p,tmp_path)
    assert code==2 and not report['metrics']
    base=tmp_path/'evaluation';base.mkdir();(tmp_path/'data').mkdir()
    shutil.copy2(p,base/p.name);shutil.copy2(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());config['maximum_voxel_mse_excess_boundary']=-1
    (base/'metrics.json').write_text(json.dumps(config))
    report,code=evaluate(p,base);assert code==2 and not report['metrics']

def test_independent_forward_metric_case():
    mask=np.zeros((192,192),bool);mask[1,1]=True;x=np.array([-1.,0.,1.])
    a={k:np.full((1,192,192,6),v) for k,v in [('amplitude',.1),('width',2.),('center',0.)]};a['field_shift']=np.zeros((1,192,192))
    z=np.zeros((192,192,3));z[1,1]=1-.6/(1+x*x)
    assert observation_rmse(a,z,x,mask)<1e-15


def test_independent_bounded_amplitude_improvement_passes(tmp_path):
    from scipy.optimize import lsq_linear
    from evaluation.judge import normalized_observations,observation_mse
    a=load_arrays(ROOT/'evaluation/baseline_reference.npz')
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as raw:z,off,mask=normalized_observations(raw)
    iy,ix=np.argwhere(mask)[len(np.argwhere(mask))//2]
    w=a['width'][0,iy,ix];c=a['center'][0,iy,ix];b=a['field_shift'][0,iy,ix]
    matrix=.25*w[None,:]**2/(.25*w[None,:]**2+(off[:,None]-b-c[None,:])**2)
    fitted=lsq_linear(matrix,1-z[iy,ix],bounds=([.02,0,0,0,0,0],[1,.5,.3,.5,.25,.25]),tol=1e-12)
    old=np.mean((matrix@a['amplitude'][0,iy,ix]-(1-z[iy,ix]))**2)
    assert np.mean((matrix@fitted.x-(1-z[iy,ix]))**2)<old-1e-10
    a['amplitude'][0,iy,ix]=fitted.x
    path=tmp_path/'joint_amplitudes.npz';np.savez(path,**a)
    report,code=evaluate(path);assert code==0,report
