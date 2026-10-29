from pathlib import Path
import numpy as np,pytest,sys,subprocess,importlib.util
R=Path(__file__).resolve().parents[2];s=importlib.util.spec_from_file_location('judge',R/'evaluation/judge.py');j=importlib.util.module_from_spec(s);s.loader.exec_module(j)
def test_known_metrics_and_degeneracy():
    assert j.cosine(np.array([1.,0]),np.array([0.,1]))==0
    np.testing.assert_allclose(j.cosine(np.array([1.,2]),np.array([2.,4])),1,atol=1e-15)
    assert j.nrmse(np.array([0.,0]),np.array([0.,2]))==np.sqrt(.5)
    with pytest.raises(ValueError):j.nrmse(np.ones(2),np.ones(2))
    with pytest.raises(ValueError):j.cosine(np.ones(2),np.zeros(2))
def test_independent_filter():
    from src.preprocessing import load_data,preprocess
    data=load_data(R/'data/raw_data.npz');_,y=preprocess(data['bold'])
    np.testing.assert_allclose(j.filtered_observations(data['bold'][None]),y,rtol=1e-12,atol=1e-12)
@pytest.mark.parametrize('kind',['key','shape','dtype','complex','nan','inf','missing','invalid'])
def test_malformed(kind,tmp_path):
    d=j.load_arrays(R/'evaluation/baseline_reference.npz');p=tmp_path/'bad.npz'
    if kind=='key':d.pop('hrf')
    if kind=='shape':d['hrf']=np.zeros(3)
    if kind=='dtype':d['hrf']=d['hrf'].astype('float32')
    if kind=='complex':d['hrf']=d['hrf'].astype('complex128')
    if kind=='nan':d['hrf'][0,0,0]=np.nan
    if kind=='inf':d['hrf'][0,0,0]=np.inf
    if kind not in ['missing','invalid']:np.savez(p,**d)
    if kind=='invalid':p.write_text('invalid',encoding='utf8')
    report,code=j.evaluate(p);assert code==2 and not report['metrics'] and report['errors']
def test_good_poor_cwd_report_failure(tmp_path):
    assert j.evaluate(R/'output/reconstruction.npz')[1]==0
    p=tmp_path/'poor.npz';np.savez(p,**{k:np.zeros(v) for k,v in j.SHAPES.items()});assert j.evaluate(p)[1]==1
    cmd=[sys.executable,str(R/'evaluation/judge.py'),'--prediction',str(R/'output/reconstruction.npz'),'--report',str(tmp_path/'new/report.json')]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    cmd[-1]=str(tmp_path);assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
def test_evaluator_failure_and_truth_priority(tmp_path,monkeypatch):
    import shutil
    e=tmp_path/'evaluation';e.mkdir();(tmp_path/'data').mkdir();shutil.copyfile(R/'evaluation/metrics.json',e/'metrics.json');shutil.copyfile(R/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    d=j.load_arrays(R/'evaluation/baseline_reference.npz');np.savez(e/'baseline_reference.npz',**d);np.savez(e/'ground_truth.npz',**{k:a*2 for k,a in d.items()});monkeypatch.setattr(j,'HERE',e)
    report,code=j.evaluate(R/'evaluation/baseline_reference.npz');assert code==1 and report['metrics']['hrf_nrmse']>0
    monkeypatch.setattr(j,'HERE',tmp_path/'missing');assert j.evaluate(R/'output/reconstruction.npz')[1]==2
