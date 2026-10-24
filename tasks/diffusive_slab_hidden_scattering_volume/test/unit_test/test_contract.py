from pathlib import Path
import json,importlib.util,subprocess,sys
import numpy as np
import pytest
from src.preprocessing import preprocess,load_observations
from src.visualization import metrics as visualization_metrics
from src.physics_model import paired_wiener
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def assets(tmp_path):
    root=tmp_path/'assets'; root.mkdir()
    (root/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes())
    (root/'baseline_reference.npz').write_bytes((ROOT/'evaluation/baseline_reference.npz').read_bytes())
    return root

def save(tmp_path,**arrays):
    p=tmp_path/'prediction.npz';np.savez_compressed(p,**arrays);return p

def test_metric_known_case():
    b=np.array([0.,1.,2.,3.]).reshape(1,1,2,2)
    a=b+1
    values=judge.measurements(a,b)
    assert values['nrmse']==pytest.approx(1/3)
    assert values['ncc']==pytest.approx(20/np.sqrt(420))
    assert values['max_error_range']==pytest.approx(1/3)
    visual=visualization_metrics(a,b)
    assert visual['ncc']==pytest.approx(20/np.sqrt(420))
    assert visual['nrmse']==pytest.approx(1/3)

def test_known_constant_wiener():
    torch.set_num_threads(1)
    result=paired_wiener(torch.ones((256,64,64)))
    torch.testing.assert_close(result,torch.full_like(result,1/(1+1/2500.)),rtol=2e-6,atol=2e-6)

@pytest.mark.parametrize('variant',['missing','key','shape','integer','complex','nan','inf','object','extra'])
def test_invalid_prediction(variant,tmp_path,assets):
    a=np.ones(judge.SHAPE,dtype=np.float32)
    if variant=='missing':p=tmp_path/'missing.npz'
    elif variant=='key':p=save(tmp_path,wrong=a)
    elif variant=='extra':p=save(tmp_path,reconstruction=a,extra=a)
    else:
        if variant=='shape':a=a[0]
        if variant=='integer':a=a.astype(np.int32)
        if variant=='complex':a=a.astype(np.complex64)
        if variant=='object':a=a.astype(object)
        if variant=='nan':a.flat[0]=np.nan
        if variant=='inf':a.flat[0]=np.inf
        p=save(tmp_path,reconstruction=a)
    report,exit=judge.evaluate(p,assets)
    assert exit==2 and report['status']=='error' and not report['metrics'] and report['errors']

def test_reference_pass(tmp_path,assets):
    r,exit=judge.evaluate(assets/'baseline_reference.npz',assets)
    assert exit==0 and r['passed'] and r['metrics']['ncc']==pytest.approx(1)

def test_zero_fails(tmp_path,assets):
    report,exit=judge.evaluate(save(tmp_path,reconstruction=np.zeros(judge.SHAPE)),assets)
    assert exit==1 and not report['passed'] and report['metrics']['ncc']==0

def test_negative_prediction_fails(tmp_path,assets):
    a=judge.load_field(assets/'baseline_reference.npz');a.flat[0]=-1e-300
    r,exit=judge.evaluate(save(tmp_path,reconstruction=a),assets)
    assert exit==1 and r['metrics']['negative_count']==1

def test_nonzero_boundary_fails(tmp_path,assets):
    a=judge.load_field(assets/'baseline_reference.npz');a[0,-1,0,0]=1e-300
    r,exit=judge.evaluate(save(tmp_path,reconstruction=a),assets)
    assert exit==1 and r['metrics']['tail_nonzero_count']==1

@pytest.mark.parametrize('variant',['missing','shape','constant','nan'])
def test_evaluator_error(variant,tmp_path,assets):
    if variant=='missing':(assets/'baseline_reference.npz').unlink()
    else:
        a=judge.load_field(assets/'baseline_reference.npz')
        if variant=='shape':a=a[0]
        if variant=='constant':a[:]=1
        if variant=='nan':a.flat[0]=np.nan
        np.savez_compressed(assets/'baseline_reference.npz',reconstruction=a)
    r,exit=judge.evaluate(save(tmp_path,reconstruction=np.zeros(judge.SHAPE)),assets)
    assert exit==2 and not r['metrics'] and r['errors']

def test_actual_truth_priority(tmp_path,assets):
    truth=judge.load_field(assets/'baseline_reference.npz')*2
    np.savez_compressed(assets/'ground_truth.npz',reconstruction=truth)
    r,exit=judge.evaluate(assets/'baseline_reference.npz',assets)
    assert exit==1 and r['metrics']['nrmse']>0

def test_malformed_truth_directory_has_priority(tmp_path,assets):
    (assets/'ground_truth.npz').mkdir()
    report,exit=judge.evaluate(assets/'baseline_reference.npz',assets)
    assert exit==2 and not report['metrics'] and report['errors']

@pytest.mark.parametrize('boundary',[-1.,float('nan'),True])
def test_invalid_threshold(boundary,tmp_path,assets):
    c=json.loads((assets/'metrics.json').read_text());c['nrmse_boundary']=boundary
    (assets/'metrics.json').write_text(json.dumps(c))
    r,exit=judge.evaluate(assets/'baseline_reference.npz',assets)
    assert exit==2 and not r['metrics']

def test_finite_numeric_overflow(tmp_path,assets):
    a=np.full(judge.SHAPE,np.finfo(float).max)
    a[:,-15:]=0
    r,exit=judge.evaluate(save(tmp_path,reconstruction=a),assets)
    assert exit==2 and not r['metrics']
    json.dumps(r,allow_nan=False)

@pytest.mark.parametrize('variant',['zero','negative','shape','nan'])
def test_input_validation(variant):
    x=np.zeros((512,32,32))
    if variant=='negative':x.flat[0]=-1
    if variant=='shape':x=x[0]
    if variant=='nan':x.flat[0]=np.nan
    with pytest.raises(ValueError):preprocess(x)

def test_raw_contract():
    x=load_observations(ROOT/'data/raw_data.npz')
    assert x.shape==(512,32,32) and x.dtype==np.float32

def test_readme_constants_and_complete_output_contract():
    from src import physics_model as physics
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for label,value in [('Scan width',physics.SCAN_WIDTH),('Effective refractive index n',physics.REFRACTIVE_INDEX),('Reduced scattering coefficient mu-s',physics.SCATTERING),('Absorption coefficient mu-a',physics.ABSORPTION),('Slab thickness',physics.THICKNESS),('Extrapolation length ze',physics.EXTRAPOLATION)]:
        line=next(line for line in text.splitlines() if line.startswith('| '+label+' |'))
        assert float(line.split('|')[2].strip())==value
    assert '(1, 512, 32, 32)' in text and '(1, 128, 32, 32)' in text
    assert 'github' not in text.lower()

def test_standard_main_full_pipeline(tmp_path):
    command=[sys.executable,'-X','utf8','-m','src.main']
    r=subprocess.run(command,cwd=ROOT,capture_output=True,encoding='utf8')
    assert r.returncode==0 and 'Full CPU reconstruction:' in r.stdout
    report,exit=judge.evaluate(ROOT/'output/reconstruction.npz')
    assert exit==0 and report['passed']

def test_cli_foreign_cwd_and_report_failure(tmp_path):
    p=ROOT/'evaluation/baseline_reference.npz';report=tmp_path/'nested/report.json'
    command=[sys.executable,'-X','utf8',str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)]
    r=subprocess.run(command,cwd=tmp_path,capture_output=True,encoding='utf8')
    assert r.returncode==0 and json.loads(report.read_text())['passed']
    blocker=tmp_path/'blocker';blocker.write_text('file')
    command[-1]=str(blocker/'report.json')
    r=subprocess.run(command,cwd=tmp_path,capture_output=True,encoding='utf8')
    assert r.returncode==2 and 'Report-write failure' in r.stderr
