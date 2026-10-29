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



def test_exact_poisson_zero_convention():
    x=np.array([0.,2.]);g=np.array([0.,1.])
    assert judge.poisson_deviance(x,g)==pytest.approx(2*(1-np.log(2)))
    assert judge.poisson_deviance(np.zeros(2),np.ones(2))>1000


@pytest.fixture
def fast_evaluator(tmp_path,monkeypatch):
    import shutil
    for name in ['metrics.json','baseline_reference.npz']:shutil.copy(ROOT/'evaluation'/name,tmp_path/name)
    monkeypatch.setattr(judge,'observation_metrics',lambda *args:{'poisson_deviance':0.,'forward_relative_L2':0.})
    return tmp_path


def test_missing_corrupt_and_negative_reference(fast_evaluator):
    root=fast_evaluator;ref=root/'baseline_reference.npz';a=judge.read_volume(ref)
    ref.unlink();assert judge.evaluate(ROOT/'output/reconstruction.npz',root)[1]==2
    a.flat[0]=-1e-20;np.savez(ref,reconstruction=a)
    assert judge.evaluate(ROOT/'output/reconstruction.npz',root)[1]==2


@pytest.mark.parametrize('key,value',[('poisson_deviance_boundary',None),('poisson_deviance_boundary',True),('poisson_deviance_boundary',float('nan')),('poisson_deviance_boundary',-1),('forward_relative_L2_boundary',-1),('negative_count_boundary',1),('negative_count_boundary',True)])
def test_invalid_boundaries(fast_evaluator,key,value):
    import json
    root=fast_evaluator;p=root/'metrics.json';c=json.loads(p.read_text());c[key]=value;p.write_text(json.dumps(c))
    r,code=judge.evaluate(ROOT/'output/reconstruction.npz',root);assert code==2 and not r['metrics']


def test_no_unsupported_truth_fallback(fast_evaluator):
    import shutil
    shutil.copy(fast_evaluator/'baseline_reference.npz',fast_evaluator/'ground_truth.npz')
    r,c=judge.evaluate(ROOT/'output/reconstruction.npz',fast_evaluator);assert c==2 and 'Unexpected actual truth' in r['errors'][0]


def test_negative_and_nonfinite_derived(fast_evaluator,monkeypatch):
    root=fast_evaluator;a=judge.read_volume(ROOT/'output/reconstruction.npz').copy();a.flat[0]=-1e-20
    p=root/'negative.npz';np.savez(p,reconstruction=a);r,c=judge.evaluate(p,root);assert c==1 and r['metrics']['negative_count']==1
    monkeypatch.setattr(judge,'observation_metrics',lambda *args:{'poisson_deviance':float('nan'),'forward_relative_L2':0.})
    r,c=judge.evaluate(ROOT/'output/reconstruction.npz',root);assert c==2 and not r['metrics']


def test_real_immutable_output_and_poor_zero(tmp_path):
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz');assert code==0,report
    p=tmp_path/'zero.npz';np.savez(p,reconstruction=np.zeros(judge.SHAPE,np.float32))
    report,code=judge.evaluate(p);assert code==1


def test_report_write_failure_and_foreign_cwd(tmp_path):
    import subprocess,sys,json
    script=ROOT/'evaluation/judge.py';report=tmp_path/'nested/report.json'
    c=subprocess.run([sys.executable,str(script),'--prediction',str(tmp_path/'absent.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,encoding='utf8',errors='replace')
    assert c.returncode==2 and json.loads(report.read_text())['metrics']=={}
    blocked=tmp_path/'blocked';blocked.write_text('file')
    c=subprocess.run([sys.executable,str(script),'--prediction',str(tmp_path/'absent.npz'),'--report',str(blocked/'report.json')],capture_output=True,encoding='utf8',errors='replace')
    assert c.returncode==2 and 'Cannot write report' in c.stderr


def test_visual_metrics_match_independent_known_cases():
    from src.visualization import metrics,projection_metrics
    x=np.array([2.,4.,8.]);r=np.array([1.,2.,4.])
    assert metrics(x,r)==judge.agreement(x,r)
    assert projection_metrics(x,r)['relative_L2']==1.
    assert projection_metrics(x,r)['poisson_nll_without_factorial']==float(np.sum(x-r*np.log(x+1e-11)))
