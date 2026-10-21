from pathlib import Path
import importlib.util,subprocess,sys,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(sp);sp.loader.exec_module(j)
def test_known_metrics_and_good_output():
    assert j.score(np.array([0.,1]),np.array([0.,1]))==dict(ncc=1.,nrmse=0.)
    assert j.score(np.array([1.,0]),np.array([0.,1]))==dict(ncc=0.,nrmse=1.)
    assert j.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==0
def test_poor_all_components(tmp_path):
    p=tmp_path/'poor.npz';np.savez_compressed(p,**{k:np.zeros(v)for k,v in j.SHAPES.items()});assert j.evaluate(p)[1]==1
def test_malformed(tmp_path):
    assert j.evaluate(tmp_path/'missing.npz')[1]==2
    for name,array in [('shape',np.zeros(3)),('integer',np.zeros(j.SHAPES['real_index'],dtype='int8')),('complex',np.zeros(j.SHAPES['real_index'],dtype='complex64')),('nan',np.full(j.SHAPES['real_index'],np.nan)),('inf',np.full(j.SHAPES['real_index'],np.inf))]:
        p=tmp_path/(name+'.npz');np.savez_compressed(p,radial=array);r,c=j.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']
    p=tmp_path/'missing_key.npz';np.savez(p,other=np.zeros(2));assert j.evaluate(p)[1]==2
def test_foreign_cwd_report_contract(tmp_path):
    p=tmp_path/'nested/report.json';cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(p)]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0;assert json.loads(p.read_text())['passed']
    cmd[-1]=str(tmp_path);r=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert r.returncode==2 and b'Report write failure'in r.stderr
def test_missing_assets_and_thresholds(tmp_path,monkeypatch):
    monkeypatch.setattr(j,'HERE',tmp_path);assert j.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==2
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz');(tmp_path/'metrics.json').write_text('{"ncc_boundary":null,"nrmse_boundary":1}')
    assert j.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==2
def test_truth_priority(tmp_path,monkeypatch):
    shutil.copyfile(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json');shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False)as f:np.savez_compressed(tmp_path/'ground_truth.npz',**{k:-f[k]for k in j.SHAPES})
    monkeypatch.setattr(j,'HERE',tmp_path);assert j.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==1
def test_public_constants():
    from src.preprocessing import THICKNESSES_MM
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    for v in THICKNESSES_MM:assert str(v)in text
def test_every_material_and_quantity_is_required(tmp_path):
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False)as f:base={k:f[k]for k in j.SHAPES}
    for k in j.SHAPES:
        for c in range(2):
            a={key:v.copy()for key,v in base.items()};a[k][0,c]=0;p=tmp_path/'bad.npz';np.savez_compressed(p,**a);assert j.evaluate(p)[1]==1
def test_degenerate_reference_rejected():
    import pytest
    with pytest.raises(ValueError):j.score(np.ones(5),np.ones(5))
def test_plot_metrics_known_cases():
    from src.visualization import metrics
    assert metrics([1,0],[0,1])==dict(ncc=0.,nrmse=1.)
