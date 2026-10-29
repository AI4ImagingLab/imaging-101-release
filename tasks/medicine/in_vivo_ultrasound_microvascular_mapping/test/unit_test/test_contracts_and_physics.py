from pathlib import Path
import importlib.util,json
import numpy as np
import pytest
from src.physics_model import regional_maximum,radial_centers,localize
from src.solvers import munkres,link_frames,smooth_track,interpolate_tracks,render_tracks,track_localizations
from src.preprocessing import original_boundary_zero,svd_filter,FRAME_RATE,BLOCK_RATE
from src.visualization import metrics


def test_boundary_indexing_and_precision():
    x=np.ones((4,5,3),dtype=np.complex64);a=original_boundary_zero(x)
    assert a.dtype==np.complex128 and np.all(x==1)
    expected=x.astype(complex);f=expected.reshape((4,-1),order='F').copy();f[:,[0,-1]]=0;f[[0,-1]]=0
    np.testing.assert_array_equal(a,f.reshape(x.shape,order='F'))
    assert a[1,0,1]==1 and a[1,0,0]==0


def test_readme_constants():
    readme=Path('README.md').read_text()
    assert '| Frame rate within a block | 500 | Hz |' in readme
    assert FRAME_RATE==500 and BLOCK_RATE==2
    assert '(1, 30, 83, 128, 200)' in readme
    assert '(1, 831, 1281)' in readme


def test_gram_projection_physics():
    t=np.arange(12);x=np.stack([np.cos(t),np.sin(t),np.cos(t/2)],axis=0).reshape(1,3,12).astype(complex)
    result=svd_filter(x,2)
    matrix=x.reshape((-1,12),order='F');u,_,_=np.linalg.svd(matrix.conj().T@matrix)
    complement=(matrix-(matrix@u[:,:1])@u[:,:1].conj().T).reshape(x.shape,order='F')
    np.testing.assert_allclose(result,complement,atol=2e-14)
    assert np.linalg.norm(result)<=np.linalg.norm(x)+1e-13


def test_regional_maximum_plateau():
    a=np.zeros((7,8));a[2:5,2:5]=1;a[2,5]=2
    mask=regional_maximum(a)
    assert mask.sum()==1 and mask[2,5]
    assert regional_maximum(np.ones((3,3))).all()


def test_radial_gaussian_and_reflection():
    z,x=np.mgrid[-2:3,-2:3];a=np.exp(-((z-.23)**2+(x+.31)**2)/(2*.9**2))
    value=radial_centers(a[None])[0]
    np.testing.assert_allclose(value,[.23,-.31],atol=.035)
    np.testing.assert_allclose(radial_centers(a[::-1,::-1][None])[0],-value,atol=1e-14)
    np.testing.assert_allclose(radial_centers((7*a)[None])[0],value,atol=1e-14)


@pytest.mark.parametrize('span',[3,4,19,20,40])
def test_smoothing_constant_and_ramp(span):
    np.testing.assert_allclose(smooth_track(np.ones(12),span),1)
    np.testing.assert_allclose(smooth_track(np.arange(12),span),np.arange(12),atol=1e-15)


def test_smoothing_known_endpoint_case():
    np.testing.assert_allclose(smooth_track(np.arange(1,10)**2,4),[1,14/3,29/3,50/3,77/3,110/3,149/3,194/3,81],rtol=0,atol=1e-14)


def test_partial_assignment_and_hard_gate():
    np.testing.assert_array_equal(munkres([[1,3,np.inf],[np.inf,np.inf,5],[np.inf,np.inf,.5]]),[0,-1,2])
    np.testing.assert_array_equal(link_frames([[0,0],[5,5]],[[1,0],[7,7]]),[0,-1])
    assert len(link_frames(np.empty((0,2)),np.ones((2,2))))==0
    np.testing.assert_array_equal(link_frames(np.ones((2,2)),np.empty((0,2))),[-1,-1])


def test_munkres_column_major_ties():
    np.testing.assert_array_equal(munkres(np.zeros((3,3))),[0,1,2])


def test_short_tracks_rejected_and_no_gap_closing():
    loc=np.column_stack([np.ones(9),np.zeros(9),np.arange(9)/10,np.arange(1,10)])
    assert len(track_localizations(loc))==1
    assert len(track_localizations(loc[:8]))==0
    changed=loc.copy();changed[4:,3]+=1
    assert not track_localizations(changed)


def test_path_speed_and_unique_crossings():
    raw=np.column_stack([np.arange(10)/10,np.zeros(10),np.arange(1,11)])
    dense=interpolate_tracks([raw])[0]
    np.testing.assert_allclose(dense[:,2],44.8,atol=1e-5)
    a=np.array([[1,2,30],[1,2,30],[1,2,30]],dtype=np.float32)
    b=np.array([[1,2,50]],dtype=np.float32)
    density,speed=render_tracks([a,b])
    assert density[19,29]==2 and speed[19,29]==40
    assert density.sum()==2 and np.count_nonzero(speed)==1


def test_metrics_known_case():
    a=np.array([1.,0]);b=np.array([1.,1.])
    assert metrics(b,a)['ncc']==pytest.approx(1/np.sqrt(2))
    assert metrics(b,a)['nrmse']==pytest.approx(1/np.sqrt(2))
    with pytest.raises(ValueError):metrics(np.ones(2),np.ones(2))


def judge_module():
    spec=importlib.util.spec_from_file_location('judge',Path('evaluation/judge.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('bad',['missing','key','shape','dtype','nan','inf','negative'])
def test_judge_rejects_malformed(tmp_path,bad):
    p=tmp_path/'prediction.npz';a=np.zeros((1,831,1281));data={'reconstruction':a.copy(),'speed':a.copy()}
    if bad=='key':data.pop('speed')
    if bad=='shape':data['speed']=np.zeros((1,2,3))
    if bad=='dtype':data['speed']=a.astype(np.float32)
    if bad=='nan':data['speed'].flat[0]=np.nan
    if bad=='inf':data['speed'].flat[0]=np.inf
    if bad=='negative':data['speed'].flat[0]=-1
    if bad!='missing':np.savez(p,**data)
    report,code=judge_module().evaluate(p)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']


def test_judge_poor_output_and_all_required_arrays(tmp_path):
    p=tmp_path/'prediction.npz';ref=dict(np.load('evaluation/baseline_reference.npz'))
    np.savez(p,reconstruction=ref['reconstruction'],speed=np.zeros_like(ref['speed']))
    report,code=judge_module().evaluate(p)
    assert code==1 and not report['passed'] and report['metrics']['speed_ncc']==0
    np.savez(p,**ref);report,code=judge_module().evaluate(p)
    assert code==0 and report['passed']


def test_judge_evaluator_error(tmp_path):
    report,code=judge_module().evaluate('missing.npz',tmp_path)
    assert code==2 and report['metrics']=={} and report['errors']


def test_judge_independent_known_metric_case(tmp_path):
    root=tmp_path/'assets';root.mkdir();config=json.loads(Path('evaluation/metrics.json').read_text());(root/'metrics.json').write_text(json.dumps(config))
    ref=np.zeros((1,831,1281));ref.flat[0]=1
    estimate=ref.copy();estimate.flat[1]=1
    np.savez(root/'baseline_reference.npz',reconstruction=ref,speed=ref)
    pred=tmp_path/'pred.npz';np.savez(pred,reconstruction=estimate,speed=estimate)
    report,code=judge_module().evaluate(pred,root)
    assert code==1
    for key in ('reconstruction','speed'):
        assert report['metrics'][key+'_ncc']==pytest.approx(1/np.sqrt(2))
        assert report['metrics'][key+'_nrmse']==pytest.approx(1/np.sqrt(ref.size))


def test_judge_cli_cwd_and_report_failure(tmp_path):
    import subprocess,sys
    judge=Path('evaluation/judge.py').resolve();reference=Path('evaluation/baseline_reference.npz').resolve()
    report=tmp_path/'nested'/'report.json'
    run=subprocess.run([sys.executable,str(judge),'--prediction',str(reference),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==0 and json.loads(report.read_text())['passed']
    fail=subprocess.run([sys.executable,str(judge),'--prediction',str(reference),'--report',str(tmp_path)],cwd=tmp_path,capture_output=True,text=True)
    assert fail.returncode==2 and 'Cannot write report' in fail.stderr


@pytest.mark.parametrize('invalid_boundary',[-1,True,False])
def test_judge_undefined_configuration_and_reference(tmp_path,invalid_boundary):
    config=json.loads(Path('evaluation/metrics.json').read_text());config['speed_nrmse_boundary']=invalid_boundary
    (tmp_path/'metrics.json').write_text(json.dumps(config))
    report,code=judge_module().evaluate('missing.npz',tmp_path)
    assert code==2 and report['metrics']=={} and 'boundary' in report['errors'][0]
    config['speed_nrmse_boundary']=.01;(tmp_path/'metrics.json').write_text(json.dumps(config))
    ref=np.ones((1,831,1281));np.savez(tmp_path/'baseline_reference.npz',reconstruction=ref,speed=ref)
    report,code=judge_module().evaluate('missing.npz',tmp_path)
    assert code==2 and report['metrics']=={} and 'Degenerate' in report['errors'][0]

def test_radial_closed_form_vs_independent_weighted_lstsq():
    from scipy.signal import convolve2d
    z,x=np.mgrid[-2:3,-2:3];image=np.exp(-((z-.23)**2+(x+.31)**2)/(2*.9**2))
    du=image[:-1,1:]-image[1:,:-1];dv=image[:-1,:-1]-image[1:,1:]
    fu=convolve2d(du,np.ones((3,3))/9,mode='same');fv=convolve2d(dv,np.ones((3,3))/9,mode='same')
    zz,xx=np.mgrid[-1.5:2:1,-1.5:2:1];g=fu**2+fv**2
    z0=(g*zz).sum()/g.sum();x0=(g*xx).sum()/g.sum();w=g/np.hypot(zz-z0,xx-x0)
    m=-(fv+fu)/(fu-fv);b=zz-m*xx;q=np.sqrt(w/(1+m*m))
    matrix=np.column_stack([(-m*q).ravel(),q.ravel()]);rhs=(b*q).ravel()
    xc,zc=np.linalg.lstsq(matrix,rhs,rcond=None)[0]
    np.testing.assert_allclose(radial_centers(image[None])[0],[zc,xc],rtol=0,atol=2e-14)
    np.testing.assert_allclose(matrix.T@(matrix@np.array([xc,zc])-rhs),0,atol=2e-14)
