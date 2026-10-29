from pathlib import Path
import importlib.util,json,shutil,sys,subprocess,os
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_hertha_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def arrays():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return {key:f[key] for key in f.files}

@pytest.fixture
def hidden(tmp_path):
    folder=tmp_path/'evaluation';folder.mkdir()
    for name in ['metrics.json','baseline_reference.npz']:shutil.copy2(ROOT/'evaluation'/name,folder/name)
    return folder

def test_known_physical_reference_matches_recorded_original_objective():
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:raw={key:f[key] for key in f.files}
    assert abs(judge.forward_objective(arrays(),raw)-29.14524246398036)<1e-8

def test_field_metrics_independent_known_case():
    reference=arrays();estimate={key:value.copy() for key,value in reference.items()}
    estimate['reconstruction'][...,0]+=1
    estimate['spin'][0,1]+=360
    estimate['ao_offsets']+=.25
    result=judge.field_metrics(estimate,reference)
    assert result['mesh_x_range_nrmse']==pytest.approx(1/np.ptp(reference['reconstruction'][...,0]))
    assert result['longitude_deg_absolute_error']==0
    assert result['ao_offsets_rms_arcsec']==pytest.approx(.25)

def test_ground_truth_precedence_and_invalid_truth_never_falls_back(hidden,tmp_path,monkeypatch):
    truth=arrays();truth['spin']=truth['spin'].copy();truth['spin'][0,0]+=.5
    prediction=tmp_path/'truth.npz';np.savez_compressed(prediction,**truth);np.savez_compressed(hidden/'ground_truth.npz',**truth)
    monkeypatch.setattr(judge,'forward_objective',lambda *a,**kw:0.)
    # Supply legitimate raw path for isolated reference selection; forward is a
    # unit-test stub only, never used for the actual physical quality gate.
    (hidden.parent/'data').mkdir();shutil.copy2(ROOT/'data/raw_data.npz',hidden.parent/'data/raw_data.npz')
    result=judge.evaluate(prediction,hidden);assert result['passed'];assert result['metrics']['latitude_deg_absolute_error']==0
    (hidden/'ground_truth.npz').write_bytes(b'not npz')
    result=judge.evaluate(prediction,hidden);assert result['status']=='error' and result['metrics']=={}

@pytest.mark.parametrize('mode',['missing','key','shape','complex','integer','nan','inf','extreme','object'])
def test_malformed_predictions_empty_metrics(hidden,tmp_path,mode):
    output=arrays();file=tmp_path/'submission.npz'
    if mode=='key':output.pop('spin')
    elif mode=='shape':output['spin']=output['spin'][0]
    elif mode=='complex':output['reconstruction']=output['reconstruction'].astype(complex)
    elif mode=='integer':output['reconstruction']=output['reconstruction'].astype(int)
    elif mode in ['nan','inf','extreme']:output['reconstruction']=np.full((1,402,3),{'nan':np.nan,'inf':np.inf,'extreme':1e300}[mode])
    elif mode=='object':output['spin']=np.array([[object(),object(),object()]],dtype=object)
    if mode!='missing':np.savez_compressed(file,**output)
    result=judge.evaluate(file,hidden);assert result['status']=='error' and result['metrics']=={} and result['errors']

@pytest.mark.parametrize('value',[True,'0.1',-.1,float('nan'),float('inf')])
def test_bad_metric_configuration(hidden,tmp_path,value):
    config=json.loads((hidden/'metrics.json').read_text(encoding='utf8'));config['boundaries']['mesh_x_range_nrmse']=value
    (hidden/'metrics.json').write_text(json.dumps(config),encoding='utf8')
    result=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',hidden)
    assert result['status']=='error' and result['metrics']=={}

@pytest.mark.parametrize('mode',['zero','initial','shuffle','wrong_spin','wrong_gain','wrong_offset'])
def test_poor_but_well_formed_arrays_fail(hidden,tmp_path,mode):
    output=arrays()
    if mode=='zero':output['reconstruction']=np.zeros_like(output['reconstruction'])
    elif mode=='initial':
        with np.load(ROOT/'test/fixtures/original_geometry.npz',allow_pickle=False) as f:output['reconstruction']=f['mesh'][None]
    elif mode=='shuffle':output['reconstruction']=output['reconstruction'][:,np.random.default_rng(7).permutation(402)]
    elif mode=='wrong_spin':output['spin']=output['spin']+np.array([[0.,0.,.01]])
    elif mode=='wrong_gain':output['ao_log_gains']=output['ao_log_gains']+.1
    else:output['occultation_offset']=output['occultation_offset']+1
    file=tmp_path/'poor.npz';np.savez_compressed(file,**output);result=judge.evaluate(file,hidden)
    assert result['status']=='fail' and not result['passed']

def test_foreign_cwd_cli_missing_and_unwritable_report(tmp_path):
    env=dict(os.environ,PYTHONIOENCODING='utf-8');command=[sys.executable,'-X','utf8',str(ROOT/'evaluation/judge.py')]
    report=tmp_path/'new/reports/judge.json'
    result=subprocess.run(command+['--prediction','absent.npz','--report',str(report)],cwd=tmp_path,env=env,capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==2 and json.loads(report.read_text(encoding='utf8'))['status']=='error'
    blocker=tmp_path/'file';blocker.write_text('sentinel',encoding='utf8')
    result=subprocess.run(command+['--prediction','absent.npz','--report',str(blocker/'report.json')],cwd=tmp_path,env=env,capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==2 and 'Cannot write' in result.stderr


def test_missing_chord_penalty_matches_unchanged_original_primitive():
    with np.load(ROOT/'test/fixtures/original_chord_fallback.npz',allow_pickle=False) as fixture:
        for chord,expected in zip(fixture['chords'],fixture['sigmoid']):
            residual,sigmoid=judge._missing_chord_residual(fixture['vertices'],chord)
            assert sigmoid==pytest.approx(expected,abs=2e-15)
            np.testing.assert_allclose(residual,[chord[0]*(1+expected),*chord[1:]],rtol=0,atol=5e-15)


def test_broken_truth_symlink_never_uses_baseline(hidden,tmp_path,monkeypatch):
    truth=hidden/'ground_truth.npz'
    # Windows denies symlink creation without its optional privilege. Emulate
    # only that filesystem predicate: broken links exist as links, not targets.
    original=Path.is_symlink
    monkeypatch.setattr(Path,'is_symlink',lambda path:True if path==truth else original(path))
    monkeypatch.setattr(judge,'forward_objective',lambda *args,**kwargs:0.)
    (hidden.parent/'data').mkdir();shutil.copy2(ROOT/'data/raw_data.npz',hidden.parent/'data/raw_data.npz')
    prediction=tmp_path/'prediction.npz';np.savez_compressed(prediction,**arrays())
    report=judge.evaluate(prediction,hidden)
    assert report['status']=='error' and report['metrics']=={}
