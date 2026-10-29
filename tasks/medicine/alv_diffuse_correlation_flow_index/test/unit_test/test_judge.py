from pathlib import Path
import subprocess,sys,importlib.util
import numpy as np,pytest
R=Path(__file__).resolve().parents[2];s=importlib.util.spec_from_file_location('judge',R/'evaluation/judge.py');j=importlib.util.module_from_spec(s);s.loader.exec_module(j)
def test_known_metrics():
    ref={k:np.ones(v) for k,v in j.SHAPES.items()};pred={k:a+2 for k,a in ref.items()}
    m=j.score(pred,ref,np.ones((390,199)),np.ones((390,199),bool))
    assert all(v==2 for v in m.values())
@pytest.mark.parametrize('kind',['key','shape','dtype','complex','nan','inf','missing','not_npz'])
def test_malformed(kind,tmp_path):
    r=j.load_arrays(R/'evaluation/baseline_reference.npz');p=tmp_path/'bad.npz'
    if kind=='key':r.pop('beta')
    if kind=='shape':r['beta']=np.ones(3)
    if kind=='dtype':r['beta']=np.ones((1,390),dtype='float32')
    if kind=='complex':r['beta']=np.ones((1,390),dtype='complex128')
    if kind=='nan':r['beta'][0,0]=np.nan
    if kind=='inf':r['beta'][0,0]=np.inf
    if kind not in ['missing','not_npz']:np.savez(p,**r)
    if kind=='not_npz':p.write_text('bad',encoding='utf8')
    report,code=j.evaluate(p);assert code==2 and not report['metrics'] and report['errors']
def test_good_poor_cwd_write(tmp_path):
    assert j.evaluate(R/'output/reconstruction.npz')[1]==0
    p=tmp_path/'poor.npz';np.savez(p,**{k:np.zeros(v) for k,v in j.SHAPES.items()});assert j.evaluate(p)[1]==1
    cmd=[sys.executable,str(R/'evaluation/judge.py'),'--prediction',str(R/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    cmd[-1]=str(tmp_path);assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
def test_assets_and_truth_precedence(tmp_path,monkeypatch):
    import shutil
    e=tmp_path/'evaluation';e.mkdir();(tmp_path/'data').mkdir();shutil.copyfile(R/'evaluation/metrics.json',e/'metrics.json');shutil.copyfile(R/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    r=j.load_arrays(R/'evaluation/baseline_reference.npz');np.savez(e/'baseline_reference.npz',**r);np.savez(e/'ground_truth.npz',**{k:a+100 for k,a in r.items()});monkeypatch.setattr(j,'HERE',e)
    report,code=j.evaluate(R/'evaluation/baseline_reference.npz');assert code==1 and report['metrics']['beta_max_abs_error']==pytest.approx(100)
    monkeypatch.setattr(j,'HERE',tmp_path/'missing');assert j.evaluate(R/'output/reconstruction.npz')[1]==2
def test_public_observation_support():
    from src.preprocessing import load_data,prepare
    weighted,mask=prepare(load_data(R/'data/raw_data.npz'))
    with np.load(R/'data/raw_data.npz') as z:w,m=j.observation_support(z)
    np.testing.assert_array_equal(w,weighted);np.testing.assert_array_equal(m,mask)
