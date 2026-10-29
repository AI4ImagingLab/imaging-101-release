from pathlib import Path
import numpy as np,json,subprocess,sys,shutil,importlib.util,pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_metrics_and_independent_operator():
    from src.solvers import configuration,outputs
    from src.physics_model import transmission
    v=np.tile(np.linspace(-4,4,512),(4,1));params=[configuration()[0] for _ in range(4)];ref=outputs(params,v)
    pred={k:x.copy() for k,x in ref.items()};pred['reconstruction'][...,0]+=.0003;pred['reconstruction'][...,1]+=.0002;pred['reconstruction'][...,2]+=.0001
    pred['optical_thickness']+=.0004;pred['baseline']+=.000005
    expected=judge.physical_spectra(v,ref['reconstruction'][0],ref['optical_thickness'][0],ref['baseline'][0])
    np.testing.assert_allclose(expected,ref['transmission'][0],atol=2e-14,rtol=0)
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());m,_=judge.score(pred,ref,v,ref['transmission'][0],cfg)
    assert m['isomer_shift_mm_per_s']==pytest.approx(.0003)
    assert m['quadrupole_splitting_mm_per_s']==pytest.approx(.0002)
    assert m['FWHM_mm_per_s']==pytest.approx(.0001)
    assert m['optical_thickness_maximum_absolute_error']==pytest.approx(.0004)
    assert m['baseline_maximum_absolute_error']==pytest.approx(.000005)
    assert m['transmission_maximum_absolute_error']==0
    assert all(m[f'observed_transmission_RMS_sample_{i}']==0 for i in range(4))

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','poor','order','overflow'])
def test_strict_contract(tmp_path,kind):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:values={k:f[k] for k in f.files}
    if kind=='key':del values['baseline']
    if kind=='shape':values['baseline']=values['baseline'][0]
    if kind=='dtype':values['baseline']=values['baseline'].astype('float32')
    if kind=='complex':values['baseline']=values['baseline'].astype('complex128')
    if kind in ['nan','inf']:values['baseline'].flat[0]=float(kind)
    if kind=='poor':values['baseline']*=1.1
    if kind=='order':values['reconstruction']=values['reconstruction'][:,:,::-1]
    if kind=='overflow':values['transmission'].fill(1e308)
    p=tmp_path/'p.npz';r=tmp_path/'sub/report.json'
    if kind!='missing':np.savez(p,**values)
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(r)],cwd=tmp_path,capture_output=True)
    report=json.loads(r.read_text());assert result.returncode==(1 if kind=='poor' else 2)
    if kind!='poor':assert report['metrics']=={} and report['errors']

def test_reference_precedence_assets_and_report_failure(tmp_path):
    shutil.copytree(ROOT/'evaluation',tmp_path/'evaluation');shutil.copytree(ROOT/'data',tmp_path/'data')
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:values={k:f[k] for k in f.files}
    np.savez(tmp_path/'p.npz',**values);values['baseline']+=1;np.savez(tmp_path/'evaluation/ground_truth.npz',**values)
    command=[sys.executable,str(tmp_path/'evaluation/judge.py'),'--prediction',str(tmp_path/'p.npz'),'--report',str(tmp_path/'r.json')]
    assert subprocess.run(command,capture_output=True).returncode==1
    (tmp_path/'evaluation/ground_truth.npz').unlink();(tmp_path/'evaluation/baseline_reference.npz').unlink()
    assert subprocess.run(command,capture_output=True).returncode==2
    result=subprocess.run(command[:-1]+[str(tmp_path)],capture_output=True)
    assert result.returncode==2 and b'Report-write failure' in result.stderr
