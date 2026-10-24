from pathlib import Path
import sys,json,importlib.util,subprocess,shutil,ast
import numpy as np,torch,pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('ebsd_independent_judge',ROOT/'evaluation/judge.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J);REF=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction']
def save(tmp,a):p=tmp/'prediction.npz';np.savez(p,reconstruction=a);return p
@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','object','zero','scaled','tiny','huge'])
def test_invalid_predictions(tmp_path,kind):
    p=tmp_path/'bad.npz';a=REF.copy()
    if kind=='key':np.savez(p,wrong=a)
    elif kind!='missing':
        if kind=='shape':a=a[0]
        if kind=='dtype':a=a.astype(np.float64)
        if kind=='complex':a=a.astype(complex)
        if kind=='nan':a[0,0,0,0]=np.nan
        if kind=='inf':a[0,0,0,0]=np.inf
        if kind=='object':a=a.astype(object)
        if kind=='zero':a*=0
        if kind=='scaled':a*=2
        if kind=='tiny':a*=np.float32(1e-20)
        if kind=='huge':a[0,0,0,0]=1e38
        np.savez(p,reconstruction=a)
    r,c=J.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']
def test_identity_field_is_valid_but_poor(tmp_path):
    a=np.zeros_like(REF);a[...,0]=1;r,c=J.evaluate(save(tmp_path,a));assert c==1 and r['status']=='fail'
def test_spatial_permutation_fails_angles(tmp_path,monkeypatch):
    monkeypatch.setattr(J,'forward_cosines',lambda q:np.ones(2000));a=REF[:,::-1].copy();r,c=J.evaluate(save(tmp_path,a));assert c==1 and r['metrics']['mean_disorientation_deg']>1
@pytest.mark.parametrize('key,value',[('mean_disorientation_deg',-1),('max_disorientation_deg',181),('max_disorientation_deg',True),('max_disorientation_deg',float('inf')),('max_disorientation_deg',float('nan')),('max_disorientation_deg','x'),('mean_forward_cosine',-1.1),('mean_forward_cosine',1.1)])
def test_invalid_limits(tmp_path,monkeypatch,key,value):
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg[key+'_boundary']=value;(tmp_path/'metrics.json').write_text(json.dumps(cfg));monkeypatch.setattr(J,'ASSETS',tmp_path);r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==2 and r['metrics']=={}
def test_truth_priority_and_invalid_truth_no_fallback(tmp_path,monkeypatch):
    assets=tmp_path/'evaluation';assets.mkdir()
    for f in ['metrics.json','baseline_reference.npz']:shutil.copy(ROOT/'evaluation'/f,assets/f)
    identity=np.zeros_like(REF);identity[...,0]=1;np.savez(assets/'ground_truth.npz',reconstruction=identity);monkeypatch.setattr(J,'ASSETS',assets);monkeypatch.setattr(J,'forward_cosines',lambda q:np.ones(2000));r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==1
    np.savez(assets/'ground_truth.npz',wrong=identity);r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==2 and r['metrics']=={}
def test_known_disorientation_sign_and_cubic_symmetry():
    q=np.array([[1.,0,0,0],[np.cos(.1),np.sin(.1),0,0]]);assert J.disorientation_degrees(q[:1],q[1:])[0]==pytest.approx(np.degrees(.2),abs=1e-12)
    np.testing.assert_allclose(J.disorientation_degrees(q,-q),0,atol=3e-6)
    g=J.cubic_symmetry();assert len(g)==24
    equiv=J.multiply(g,q[1]);np.testing.assert_allclose(J.disorientation_degrees(equiv,np.repeat(q[1:],24,0)),0,atol=3e-6)
def test_physical_left_right_action_not_equivalent():
    q=np.array([.7,.2,.3,.4]);q/=np.linalg.norm(q);g=np.array([np.sqrt(.5),np.sqrt(.5),0,0]);v=np.array([[.2,.3,.7],[.4,-.8,.1]]);v/=np.linalg.norm(v,axis=-1,keepdims=True)
    intensity=lambda r:np.sum(r**4,axis=-1)  # Exactly invariant to proper cubic axes.
    base=intensity(J.rotate(q,v));left=intensity(J.rotate(J.multiply(g,q),v));right=intensity(J.rotate(J.multiply(q,g),v));np.testing.assert_allclose(left,base,atol=1e-14);assert np.max(abs(right-base))>.01
def test_independent_numpy_preprocessing_and_geometry():
    obs,master,rays=J.observation_assets()
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as z:
        p=z['cleaned_patterns'].reshape(2000,900);p-=p.mean(-1,keepdims=True);np.testing.assert_allclose(obs,p,atol=4e-7);np.testing.assert_allclose(master,z['master_processed'],atol=4e-7);r=z['detector_rays'].astype(float);r/=np.linalg.norm(r,axis=-1,keepdims=True);np.testing.assert_allclose(rays,r,atol=2e-7)
def test_forward_cosine_is_physical_symmetry_and_sign_invariant():
    assets=J.observation_assets();q=REF.reshape(-1,4)[:5].astype(float);g=J.cubic_symmetry()[7];a=J.forward_cosines(q,assets);b=J.forward_cosines(J.multiply(g,q),assets);c=J.forward_cosines(-q,assets);np.testing.assert_allclose(a,b,atol=1e-12);np.testing.assert_allclose(a,c,atol=1e-12)
def test_foreign_cwd_and_report_write_failure(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')];p=subprocess.run(command,cwd=tmp_path,capture_output=True);assert p.returncode==0 and json.loads((tmp_path/'nested/report.json').read_text())['passed'];command[-1]=str(tmp_path);p=subprocess.run(command,cwd=tmp_path,capture_output=True);assert p.returncode==2 and b'Cannot write report' in p.stderr
def test_evaluator_missing_assets(tmp_path,monkeypatch):
    monkeypatch.setattr(J,'ASSETS',tmp_path);r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==2 and r['metrics']=={}
def test_readme_metadata_no_leak_and_local_backend():
    for name in ['README.md','requirements.txt']:
        t=(ROOT/name).read_text(encoding='utf-8').lower();assert not any(k in t for k in ['github.com','github.io','githubusercontent','codeload','baseline_reference.npz','ground_truth.npz'])
    text=(ROOT/'README.md').read_text(encoding='utf-8');assert '(1,40,50,4)' in text and '0.4221,0.2179,0.4954' in text and '149-row' in text and '60 by60' in text
    import matplotlib
    before=matplotlib.get_backend();import src.main,src.visualization
    assert matplotlib.get_backend()==before
    for path in (ROOT/'src').glob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'));names=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):names.extend(a.name for a in n.names)
            if isinstance(n,ast.ImportFrom):names.append(n.module or '')
        assert not any(s.split('.')[0] in ['ebsdtorch','kikuchipy','orix','diffsims','skimage'] for s in names)
