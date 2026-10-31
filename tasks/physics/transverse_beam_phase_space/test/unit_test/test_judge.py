from pathlib import Path
import json,subprocess,sys,importlib.util,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def good():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def save(tmp,data):
    p=tmp/'prediction.npz';np.savez_compressed(p,**data);return p

def run(tmp,p,home=ROOT):
    out=tmp/'report/sub/report.json'
    result=subprocess.run([sys.executable,str(home/'evaluation/judge.py'),'--prediction',str(p),'--report',str(out)],cwd=tmp,capture_output=True,text=True,encoding='utf-8')
    assert out.exists(),(result.returncode,result.stdout,result.stderr)
    return result.returncode,json.loads(out.read_text(encoding='utf-8'))

def test_full_standard_cli_and_foreign_cwd(tmp_path):
    process=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    assert process.returncode==0,process.stderr
    code,report=run(tmp_path,ROOT/'output/reconstruction.npz')
    assert code==0 and report['passed'] and set(report['metrics'])==set(judge.NAMES)

def test_float32_and_equivalent_finite_cell_reflection(tmp_path):
    a=good()
    assert run(tmp_path,save(tmp_path,{k:v.astype(np.float32) for k,v in a.items()}))[0]==0
    # Exact signed-axis transformation changes both outputs. It expresses the
    # identical physical density using a different finite grid representation.
    a['reconstruction']=a['reconstruction'][:,::-1].copy()
    a['normalization'][:,0,:]*=-1
    code,r=run(tmp_path,save(tmp_path,a))
    assert code==0 and r['passed']

@pytest.mark.parametrize('kind',['missing','key','shape_mass','shape_matrix','integer','complex','nan_mass','inf_matrix','extra'])
def test_malformed_prediction(tmp_path,kind):
    a=good()
    if kind=='missing':p=tmp_path/'absent.npz'
    else:
        if kind=='key':a.pop('normalization')
        if kind=='extra':a['other']=np.array([1.])
        if kind=='shape_mass':a['reconstruction']=a['reconstruction'][0]
        if kind=='shape_matrix':a['normalization']=a['normalization'][0]
        if kind=='integer':a['reconstruction']=a['reconstruction'].astype(np.int32)
        if kind=='complex':a['normalization']=a['normalization'].astype(np.complex128)
        if kind=='nan_mass':a['reconstruction'].flat[0]=np.nan
        if kind=='inf_matrix':a['normalization'].flat[0]=np.inf
        p=save(tmp_path,a)
    code,r=run(tmp_path,p)
    assert code==2 and r['status']=='error' and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('kind',['zero','scale','negative','gaussian','spike','bad_basis','singular_basis','tiny_full_rank_basis'])
def test_bad_scientific_predictions(tmp_path,kind):
    a=good();p=a['reconstruction']
    if kind=='zero':p*=0
    if kind=='scale':p*=2
    if kind=='negative':p.flat[0]=-.01
    if kind=='gaussian':
        axis=(np.linspace(-4,4,36)[:-1]+np.linspace(-4,4,36)[1:])/2
        centers=np.stack(np.meshgrid(*[axis]*4,indexing='ij'),axis=-1)
        p[0]=np.exp(-.5*np.sum(centers**2,axis=-1));p/=p.sum()
    if kind=='spike':p*=0;p[0,17,17,17,17]=1
    if kind=='bad_basis':a['normalization']=np.eye(4)[None]
    if kind=='singular_basis':a['normalization']*=0
    if kind=='tiny_full_rank_basis':
        a['normalization']=np.eye(4)[None]*1e-30
        assert np.linalg.matrix_rank(a['normalization'][0])==4
    code,r=run(tmp_path,save(tmp_path,a))
    assert code==1 and not r['passed'] and r['status']=='fail'

@pytest.mark.parametrize('value',[True,False,'1',None,[],[1],float('nan'),float('inf'),-1])
def test_bad_metric_numeric_config(tmp_path,value):
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf-8'))
    cfg['relative_entropy_KL_boundary']=value
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg),encoding='utf-8')
    with pytest.raises(ValueError):judge.configuration(path)

@pytest.mark.parametrize('mode',['empty','missing','extra','comparison'])
def test_bad_metric_set(tmp_path,mode):
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf-8'))
    if mode=='empty':cfg={}
    if mode=='missing':cfg.pop('profile_23_relative_l2_boundary')
    if mode=='extra':cfg['unknown_boundary']=1
    if mode=='comparison':cfg['comparisons']={}
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg),encoding='utf-8')
    with pytest.raises(ValueError):judge.configuration(path)

def test_evaluator_and_report_failure(tmp_path):
    isolated=tmp_path/'evaluator';(isolated/'evaluation').mkdir(parents=True)
    for name in ['judge.py','metrics.json']:shutil.copyfile(ROOT/'evaluation'/name,isolated/'evaluation'/name)
    code,r=run(tmp_path,save(tmp_path,good()),isolated)
    assert code==2 and not r['metrics']
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(ROOT/'README.md'/'not_a_dir.json')],capture_output=True,text=True,encoding='utf-8')
    assert p.returncode==2 and 'Cannot write' in p.stderr

def test_independent_voxel_analytic_cases_and_covariance():
    f=np.load(ROOT/'test/fixtures/analytic_voxel_cases.npz',allow_pickle=False)
    coords=f['coords'];dx=coords[1]-coords[0]
    one=judge.uniform_voxel_projection(np.ones(1),np.zeros((1,4)),np.array([1.,0,0,0]),coords,64)
    np.testing.assert_allclose(one,f['one_uniform_exact'],rtol=1e-9,atol=1e-9)
    four=judge.uniform_voxel_projection(np.ones(1),np.zeros((1,4)),np.array([1.,.5,.3,.2]),coords,64)
    expected=(8/35)**2/12*(1+.5**2+.3**2+.2**2)
    assert abs(np.sum(coords**2*four)*dx/expected-1)<.001
    assert abs(four.sum()*dx-1)<1e-12
    np.testing.assert_allclose(four,four[::-1],rtol=1e-9,atol=1e-9)
    from src.preprocessing import load_observations,preprocess
    from src.physics_model import fit_covariance
    raw=load_observations(ROOT/'data/raw_data.npz');m,v=preprocess(raw)
    a,T,S=judge.observations(ROOT/'data/raw_data.npz')
    np.testing.assert_allclose(a,m,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(S,fit_covariance(v,T),rtol=1e-9,atol=1e-10)
