from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch
from src.preprocessing import load_data,scatter_estimate,WINDOW_BOUNDS
from src.physics_model import detection_probability,rotation_grid,rotate_volume,gaussian_kernel,blur_volume

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def test_raw_shapes_and_constants():
    d=load_data(ROOT/'data/raw_data.npz')
    assert d['energy_windows'].shape==(3,96,128,128)
    assert d['angles'][0]==np.float32(3.7144622802734375)
    assert np.all(np.diff(d['angles'])>0)
    assert WINDOW_BOUNDS[0]==(187.19999694824,228.80000305176)


def test_half_voxel_beer_lambert():
    mu=torch.full((11,3,2),.12)
    exact=np.exp(-.12*.47951998710631993*(10.5-np.arange(11)))[:,None,None]
    assert np.max(abs(detection_probability(mu).numpy()-exact))<1e-7


def test_zero_attenuation():
    assert torch.equal(detection_probability(torch.zeros(7,3,4)),torch.ones(7,3,4))


def test_rotation_geometry_independent_scipy():
    from scipy.ndimage import map_coordinates
    rng=np.random.default_rng(1804);x=rng.normal(size=(29,29));angle=31.25
    theta=angle*float(np.float32(np.pi))/180;c=np.cos(theta);s=np.sin(theta)
    M=np.array([[c,s,(1-c)*14-s*14],[-s,c,s*14+(1-c)*14],[0,0,1.]])
    N=np.array([[np.float32(2/28),0,-1],[0,np.float32(2/28),-1],[0,0,1.]])
    inv=np.linalg.inv(N@(M@np.linalg.inv(N)))
    yy,xx=np.meshgrid(np.linspace(-1,1,29),np.linspace(-1,1,29),indexing='ij')
    coord=inv@np.stack([xx,yy,np.ones_like(xx)]).reshape(3,-1)
    expected=map_coordinates(x,[(coord[1].reshape(29,29)+1)*14,(coord[0].reshape(29,29)+1)*14],order=1,mode='grid-constant',prefilter=False)
    got=rotate_volume(torch.tensor(x[:,:,None]),rotation_grid(angle,29,29,'cpu',torch.float64)).numpy()[:,:,0]
    assert np.max(abs(got-expected))<5e-13


def test_psf_kernel_physical_width_and_normalization():
    k=gaussian_kernel(22.3,182).numpy()[:,0]
    d=np.maximum(22.3+(90.5-np.arange(182))*.47951998710631993,0)
    sig=np.sqrt((.03161982752665455*d+.1248503046423388)**2+.16137114205472364**2)/.47951998710631993
    x=np.arange(-(k.shape[1]//2),k.shape[1]//2+1)
    independent=np.exp(-x[None,:]**2/(2*sig[:,None]**2+1e-11));independent/=independent.sum(1,keepdims=True)
    assert np.max(abs(k-independent))<2e-7
    assert np.max(abs(k.sum(1)-1))<2e-7


def test_separable_blur_has_symmetric_discrete_adjoint():
    torch.manual_seed(14);x=torch.rand(13,15,17);y=torch.rand_like(x);k=gaussian_kernel(2,13)
    lhs=torch.sum(blur_volume(x,k).double()*y.double());rhs=torch.sum(x.double()*blur_volume(y,k).double())
    assert abs(float(lhs-rhs))/float(abs(lhs))<1e-7


def test_metric_known_cosine_and_dynamic_range():
    r=np.array([1,2,4],dtype=np.float32);x=np.array([2,4,8],dtype=np.float32)
    m=judge.agreement(x,r);assert abs(m['ncc']-1)<1e-15
    assert abs(m['nrmse']-np.sqrt(7)/3)<1e-15


@pytest.mark.parametrize('kind',['missing','key','shape','float64','integer','complex','nan','inf','object'])
def test_bad_predictions(tmp_path,kind):
    path=tmp_path/'input.npz';a=np.zeros(judge.SHAPE,np.float32)
    if kind=='missing':pass
    elif kind=='key':np.savez(path,wrong=a)
    else:
        if kind=='shape':a=a[0]
        elif kind=='float64':a=a.astype('float64')
        elif kind=='integer':a=a.astype('int16')
        elif kind=='complex':a=a.astype('complex64')
        elif kind=='nan':a.flat[0]=np.nan
        elif kind=='inf':a.flat[0]=np.inf
        elif kind=='object':a=np.array([object()],dtype=object)
        np.savez(path,reconstruction=a)
    report,code=judge.evaluate(path)
    assert code==2 and report['metrics']=={} and report['errors']


def test_ground_truth_priority_and_invalid_truth_no_fallback(tmp_path,monkeypatch):
    ref=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction']
    np.savez(tmp_path/'baseline_reference.npz',reconstruction=ref)
    np.savez(tmp_path/'ground_truth.npz',reconstruction=ref*2)
    (tmp_path/'metrics.json').write_text((ROOT/'evaluation/metrics.json').read_text())
    pred=tmp_path/'prediction.npz';np.savez(pred,reconstruction=ref)
    monkeypatch.setattr(judge,'observation_error',lambda *args:0.)
    report,code=judge.evaluate(pred,tmp_path)
    expected=np.sqrt(np.mean(ref.astype(float)**2))/(2*np.ptp(ref))
    assert code==1 and abs(report['metrics']['nrmse']-expected)<1e-15
    np.savez(tmp_path/'ground_truth.npz',reconstruction=ref.astype('float64'))
    report,code=judge.evaluate(pred,tmp_path);assert code==2 and report['metrics']=={}


def test_missing_evaluator_reference(tmp_path):
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and report['metrics']=={}


def test_zero_and_constant_fail(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'observation_error',lambda *args:.9)
    for value in [0.,1.]:
        pred=tmp_path/'p.npz';np.savez(pred,reconstruction=np.full(judge.SHAPE,value,np.float32))
        r,code=judge.evaluate(pred);assert code==1 and r['status']=='fail'


def test_invalid_metric_boundary(tmp_path):
    import shutil,json
    shutil.copy(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    m=json.loads((ROOT/'evaluation/metrics.json').read_text());m['nrmse_boundary']=None
    (tmp_path/'metrics.json').write_text(json.dumps(m))
    r,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and r['metrics']=={}


def test_visual_metrics_match_independent_known_cases():
    from src.visualization import metrics,projection_metrics
    x=np.array([2.,4.,8.]);r=np.array([1.,2.,4.])
    assert metrics(x,r)==judge.agreement(x,r)
    assert projection_metrics(x,r)['relative_L2']==1.
    expected=float(np.sum(x-r*np.log(x+1e-11)))
    assert projection_metrics(x,r)['poisson_nll_without_factorial']==expected


def test_cli_report_write_failure(tmp_path):
    import subprocess,sys
    blocked=tmp_path/'blocked';blocked.write_text('ordinary file')
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'absent.npz'),'--report',str(blocked/'report.json')],capture_output=True,text=True)
    assert result.returncode==2 and 'Cannot write report' in result.stderr


def test_cli_missing_input_creates_nested_report(tmp_path):
    import subprocess,sys,json
    target=tmp_path/'nested/report.json'
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'absent.npz'),'--report',str(target)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2
    r=json.loads(target.read_text());assert r['metrics']=={} and r['errors']


def test_corrupt_raw_rejected(tmp_path):
    path=tmp_path/'raw.npz';d=load_data(ROOT/'data/raw_data.npz')
    np.savez(path,**{k:v[None] for k,v in d.items() if k!='radii'})
    with pytest.raises(KeyError):load_data(path)


def test_one_tiny_negative_voxel_is_valid_quality_failure(tmp_path):
    a=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'].copy()
    a.flat[np.flatnonzero(a.ravel()==0)[0]]=np.float32(-1e-20)
    pred=tmp_path/'negative.npz';np.savez(pred,reconstruction=a)
    report,code=judge.evaluate(pred)
    assert code==1 and report['status']=='fail' and report['metrics']['negative_count']==1
    assert report['metrics']['nrmse']<1e-6 and report['errors']==[]


@pytest.mark.parametrize('key,value',[('ncc_boundary',0),('ncc_boundary',-.1),('ncc_boundary',1.01),('ncc_boundary',True),('nrmse_boundary',-.01),('max_error_over_reference_range_boundary',-1),('forward_relative_L2_boundary',-.1),('negative_count_boundary',1),('negative_count_boundary',True)])
def test_metric_boundary_domain_errors(tmp_path,key,value):
    import shutil,json
    shutil.copy(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg[key]=value
    (tmp_path/'metrics.json').write_text(json.dumps(cfg))
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and report['metrics']=={} and report['errors']


def test_finite_extreme_prediction_overflow_has_writable_error_json(tmp_path):
    import json
    a=np.full(judge.SHAPE,np.finfo(np.float32).max,np.float32)
    path=tmp_path/'finite_extreme.npz';np.savez_compressed(path,reconstruction=a)
    report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={}
    assert 'Non-finite measured metric' in report['errors'][0]
    encoded=json.dumps(report,allow_nan=False);assert json.loads(encoded)['passed'] is False


def test_nonfinite_derived_metric_never_leaks_json(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(judge,'observation_error',lambda *args:float('nan'))
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz')
    assert code==2 and report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)


def test_negative_hidden_reference_is_evaluator_error(tmp_path):
    import shutil
    a=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'].copy();a.flat[0]=-1e-20
    np.savez(tmp_path/'baseline_reference.npz',reconstruction=a)
    shutil.copy(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and report['metrics']=={} and 'negative evaluator reference' in report['errors'][0]
