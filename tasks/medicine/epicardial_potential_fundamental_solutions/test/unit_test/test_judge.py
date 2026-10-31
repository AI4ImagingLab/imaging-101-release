from pathlib import Path
import importlib.util,subprocess,sys
import numpy as np,pytest
R=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('judge',R/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def test_known_metrics():
    m=j.score(np.array([2.,0.]),np.array([0.,2.]));assert m=={'ncc':0.,'nrmse':1.,'max_abs_error':2.}
    with pytest.raises(ValueError):j.score(np.ones(2),np.ones(2))
@pytest.mark.parametrize('kind',['key','shape','dtype','complex','nan','inf','missing','not_npz'])
def test_malformed(kind,tmp_path):
    r={'reconstruction':j.load_arrays(R/'evaluation/baseline_reference.npz')};p=tmp_path/'bad.npz'
    if kind=='key':r={}
    if kind=='shape':r['reconstruction']=np.ones(247)
    if kind=='dtype':r['reconstruction']=r['reconstruction'].astype('float32')
    if kind=='complex':r['reconstruction']=r['reconstruction'].astype('complex128')
    if kind=='nan':r['reconstruction'][0,0]=np.nan
    if kind=='inf':r['reconstruction'][0,0]=np.inf
    if kind not in ['missing','not_npz']:np.savez(p,**r)
    if kind=='not_npz':p.write_text('bad',encoding='utf8')
    report,code=j.evaluate(p);assert code==2 and not report['metrics'] and report['errors']
def test_good_poor_cwd_write(tmp_path):
    assert j.evaluate(R/'output/reconstruction.npz')[1]==0
    p=tmp_path/'poor.npz';np.savez(p,reconstruction=np.zeros((1,247)));assert j.evaluate(p)[1]==1
    cmd=[sys.executable,str(R/'evaluation/judge.py'),'--prediction',str(R/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    cmd[-1]=str(tmp_path);assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
def test_assets_and_truth_precedence(tmp_path,monkeypatch):
    import shutil
    e=tmp_path/'evaluation';e.mkdir();shutil.copyfile(R/'evaluation/metrics.json',e/'metrics.json')
    r=j.load_arrays(R/'evaluation/baseline_reference.npz');np.savez(e/'baseline_reference.npz',reconstruction=r);np.savez(e/'ground_truth.npz',reconstruction=r+100);monkeypatch.setattr(j,'HERE',e)
    report,code=j.evaluate(R/'evaluation/baseline_reference.npz');assert code==1 and report['metrics']['max_abs_error']==pytest.approx(100)
    monkeypatch.setattr(j,'HERE',tmp_path/'missing');assert j.evaluate(R/'output/reconstruction.npz')[1]==2
