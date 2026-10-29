"""Independent known-number and strict evaluator contract cases; no solver import."""
from pathlib import Path
import importlib.util,json,shutil,subprocess,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('coastal_judge',ROOT/'evaluation/judge.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)

def native():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in ['reconstruction','validity']}

def save(tmp_path,arrays):
    p=tmp_path/'prediction.npz';np.savez(p,**arrays);return p

def test_independent_metric_case():
    y=np.array([2.,4.,6.]);ncc,error=J.score(y*.5,y)
    assert ncc==pytest.approx(1.,abs=1e-14)
    assert error==pytest.approx(np.sqrt(14/3)/4,abs=1e-14)
    assert J.score(y,y)==pytest.approx((1.,0.),abs=1e-14)
    assert J.score(-y,y)[0]==pytest.approx(-1.)
    assert J.score(np.zeros(3),y)[0]==0
    assert np.isfinite(J.score(np.full(3,np.finfo(float).max),y)).all()
    with pytest.raises(ValueError):J.score(y,np.ones(3))

def test_native_and_truth_priority(tmp_path):
    a=native();report=J.evaluate(save(tmp_path,a));assert report['passed'] and report['metrics']['interior_nrmse']>0
    with np.load(ROOT/'evaluation/ground_truth.npz',allow_pickle=False) as z:a['reconstruction']=z['reconstruction'].copy()
    a['reconstruction'][~a['validity']]=0
    report=J.evaluate(save(tmp_path,a));assert report['passed'] and report['metrics']['interior_nrmse']==0
    assert report['metrics']['required_coverage']==1

@pytest.mark.parametrize('kind',['zero','wrong_sign','vertical_offset','selective_coverage','bad_extra_sites'])
def test_valid_poor_predictions_fail(tmp_path,kind):
    a=native();v=a['validity']
    if kind=='zero':a['reconstruction'][:]=0;a['validity'][:]=False
    elif kind=='wrong_sign':a['reconstruction']*=-1
    elif kind=='vertical_offset':a['reconstruction'][v]+=1.
    elif kind=='selective_coverage':
        selected=np.flatnonzero(v[0])[10:];a['validity'][0,selected]=False;a['reconstruction'][0,selected]=0
    else:a['reconstruction'][~v]=10000.;a['validity'][:]=True
    r=J.evaluate(save(tmp_path,a));assert r['status']=='fail' and not r['passed'] and not r['errors']
    assert all(np.isfinite(x) for x in r['metrics'].values())

def test_additional_valid_sites_are_scored(tmp_path):
    a=native()
    with np.load(ROOT/'evaluation/ground_truth.npz',allow_pickle=False) as z:a['reconstruction']=z['reconstruction'].copy()
    a['validity'][:]=True;r=J.evaluate(save(tmp_path,a))
    assert r['passed'] and r['metrics']['all_valid_nrmse']==0

@pytest.mark.parametrize('kind',['missing_key','extra','shape','float32','complex','object','nan','inf','mask_integer','bad_sentinel'])
def test_malformed_predictions_error(tmp_path,kind):
    a=native()
    if kind=='missing_key':a.pop('validity')
    elif kind=='extra':a['extra']=np.ones(1)
    elif kind=='shape':a['reconstruction']=a['reconstruction'][0]
    elif kind in ['float32','complex','object']:a['reconstruction']=a['reconstruction'].astype({'float32':np.float32,'complex':np.complex128,'object':object}[kind])
    elif kind in ['nan','inf']:a['reconstruction'][0,0]=np.nan if kind=='nan' else np.inf
    elif kind=='mask_integer':a['validity']=a['validity'].astype(np.uint8)
    else:a['reconstruction'][~a['validity']]=1
    r=J.evaluate(save(tmp_path,a));assert r['status']=='error' and not r['passed'] and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('kind',['missing_truth','bad_truth','bad_config','missing_footprint'])
def test_evaluator_failures(tmp_path,kind):
    assets=tmp_path/'assets';assets.mkdir()
    for name in ['ground_truth.npz','baseline_reference.npz','metrics.json']:shutil.copyfile(ROOT/'evaluation'/name,assets/name)
    if kind=='missing_truth':(assets/'ground_truth.npz').unlink()
    elif kind=='missing_footprint':(assets/'baseline_reference.npz').unlink()
    elif kind=='bad_truth':np.savez(assets/'ground_truth.npz',reconstruction=np.zeros((1,8963)))
    else:
        p=assets/'metrics.json';c=json.loads(p.read_text());c['interior_ncc_boundary']=float('nan');p.write_text(json.dumps(c))
    r=J.evaluate(save(tmp_path,native()),assets);assert r['status']=='error' and r['metrics']=={} and r['errors']

def test_missing_and_corrupt_npz(tmp_path):
    p=tmp_path/'absent.npz';assert J.evaluate(p)['status']=='error'
    p.write_bytes(b'not an NPZ');assert J.evaluate(p)['metrics']=={}

def test_cli_exit_codes_foreign_cwd_and_report_failure(tmp_path):
    prediction=save(tmp_path,native());report=tmp_path/'nested/report.json';judge=ROOT/'evaluation/judge.py'
    def run(path=report):return subprocess.run([sys.executable,str(judge),'--prediction',str(prediction),'--report',str(path)],cwd=tmp_path,capture_output=True)
    assert run().returncode==0 and json.loads(report.read_text())['status']=='pass'
    a=native();a['reconstruction']*=-1;save(tmp_path,a);assert run().returncode==1
    prediction.write_bytes(b'bad');assert run().returncode==2 and json.loads(report.read_text())['metrics']=={}
    save(tmp_path,native());blocking=tmp_path/'blocking';blocking.write_text('file')
    result=run(blocking/'report.json');assert result.returncode==2 and b'Report write failed' in result.stderr


def test_observation_only_interior_geometry():
    # Independent direct distances, not a cleaned solver or a GT-error mask.
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:xy=z['reconstruction_xy'][0]
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:required=z['validity'][0];interior=z['interior_validity'][0]
    expected=np.zeros(len(xy),dtype=bool);outside=xy[~required]
    for start in range(0,len(xy),128):
        delta=xy[start:start+128,None,:]-outside[None,:,:]
        expected[start:start+128]=np.min(np.sum(delta*delta,axis=2),axis=1)>20.**2
    expected&=required
    assert np.array_equal(expected,interior) and required.sum()==5615 and interior.sum()==4410


@pytest.mark.parametrize('where,count,passes',[('edge',100,True),('edge',113,False),('interior',1,False)])
def test_boundary_tolerance_does_not_allow_interior_selection(tmp_path,where,count,passes):
    a=native()
    with np.load(ROOT/'evaluation/ground_truth.npz',allow_pickle=False) as z:a['reconstruction']=z['reconstruction'].copy()
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:inside=z['interior_validity'][0]
    candidates=np.flatnonzero(inside if where=='interior' else a['validity'][0]&~inside)
    a['validity'][0,candidates[:count]]=False;a['reconstruction'][~a['validity']]=0
    r=J.evaluate(save(tmp_path,a));assert r['passed']==passes and r['status']==('pass' if passes else 'fail')
    if where=='interior':assert r['metrics']['interior_coverage']<1 and r['metrics']['interior_nrmse']==np.finfo(float).max


@pytest.mark.parametrize('kind',['missing_interior','interior_dtype','interior_outside','boolean_boundary'])
def test_malformed_coverage_assets(tmp_path,kind):
    assets=tmp_path/'assets';assets.mkdir()
    for name in ['ground_truth.npz','baseline_reference.npz','metrics.json']:shutil.copyfile(ROOT/'evaluation'/name,assets/name)
    p=assets/'baseline_reference.npz'
    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    if kind=='missing_interior':a.pop('interior_validity')
    elif kind=='interior_dtype':a['interior_validity']=a['interior_validity'].astype(np.uint8)
    elif kind=='interior_outside':
        inside=np.flatnonzero(a['interior_validity'][0])[0];outside=np.flatnonzero(~a['validity'][0])[0]
        a['interior_validity'][0,inside]=False;a['interior_validity'][0,outside]=True
    else:
        q=assets/'metrics.json';c=json.loads(q.read_text());c['required_coverage_boundary']=True;q.write_text(json.dumps(c))
    np.savez(p,**a)
    r=J.evaluate(save(tmp_path,native()),assets);assert r['status']=='error' and r['metrics']=={}
