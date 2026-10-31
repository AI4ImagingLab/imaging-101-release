from pathlib import Path
import importlib.util,json,shutil,subprocess,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('qens_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def bundle(tmp_path):
    evaluation=tmp_path/'evaluation';evaluation.mkdir();(tmp_path/'data').mkdir()
    for f in ['metrics.json','baseline_reference.npz']:shutil.copy2(ROOT/'evaluation'/f,evaluation/f)
    shutil.copy2(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    prediction=tmp_path/'prediction.npz';shutil.copy2(ROOT/'output/reconstruction.npz',prediction)
    return evaluation,prediction

def test_fresh_pass_and_independent_closed_bessel_operator(bundle):
    e,p=bundle;r,c=judge.evaluate(p,e);assert c==0 and r['passed'];assert r['metrics']['weighted_sse']<5661.294967552445

@pytest.mark.parametrize('value',[True,False,'1',[],{},None,0,-1,float('nan'),float('inf'),float('-inf')])
@pytest.mark.parametrize('key',list(judge.THRESHOLDS))
def test_malformed_boundaries(bundle,value,key):
    e,p=bundle;config=json.loads((e/'metrics.json').read_text());config[key+'_boundary']=value;(e/'metrics.json').write_text(json.dumps(config))
    r,c=judge.evaluate(p,e);assert c==2 and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('key',list(judge.THRESHOLDS))
def test_missing_boundary(bundle,key):
    e,p=bundle;config=json.loads((e/'metrics.json').read_text());del config[key+'_boundary'];(e/'metrics.json').write_text(json.dumps(config));r,c=judge.evaluate(p,e);assert c==2 and not r['metrics']

@pytest.mark.parametrize('kind',['missing_key','extra_key','shape','complex','float32','object','nan','inf','bool','negative_bound','upper_bound'])
def test_malformed_predictions(bundle,kind):
    e,p=bundle
    with np.load(p) as f:a={k:f[k] for k in f.files}
    if kind=='missing_key':del a['predicted_197']
    elif kind=='extra_key':a['extra']=np.zeros(1)
    elif kind=='shape':a['reconstruction']=a['reconstruction'][0]
    elif kind in ['complex','float32','object','bool']:a['reconstruction']=a['reconstruction'].astype({'complex':complex,'float32':np.float32,'object':object,'bool':bool}[kind])
    elif kind in ['nan','inf']:a['predicted_360'][0,0,0]=float(kind)
    elif kind=='negative_bound':a['reconstruction'][0,4]=-.1
    else:a['reconstruction'][0,0]=1e100
    np.savez(p,**a);r,c=judge.evaluate(p,e);assert c==2 and r['metrics']=={}

def test_valid_poor_fit_rejected(bundle):
    e,p=bundle
    with np.load(p) as f:a={k:f[k] for k in f.files}
    a['predicted_360']+=1;np.savez(p,**a);r,c=judge.evaluate(p,e);assert c==1 and not r['passed'] and r['metrics']['curve_360_relative']>1e-8

def test_each_output_quantity_is_scored(bundle):
    e,p=bundle
    with np.load(p) as f:original={k:f[k] for k in f.files}
    for key in original:
        a={k:v.copy() for k,v in original.items()}
        if key=='reconstruction':a[key][0,2]*=1.1
        else:a[key]+=1e-3
        np.savez(p,**a);assert judge.evaluate(p,e)[1]==1

def test_truth_precedes_baseline_and_missing_assets_error(bundle):
    e,p=bundle;shutil.copy2(p,e/'ground_truth.npz');(e/'baseline_reference.npz').write_bytes(b'bad')
    assert judge.evaluate(p,e)[1]==0
    (e/'ground_truth.npz').write_bytes(b'bad');r,c=judge.evaluate(p,e);assert c==2 and not r['metrics']
    (e/'ground_truth.npz').unlink();(e/'baseline_reference.npz').unlink();assert judge.evaluate(p,e)[1]==2

def test_missing_prediction_and_bad_calibration(bundle):
    e,p=bundle;assert judge.evaluate(p.with_name('missing.npz'),e)[1]==2
    with np.load(e.parent/'data/raw_data.npz') as f:a={k:f[k] for k in f.files}
    a['acquisition_mask_360'][0,0,0,0]=not a['acquisition_mask_360'][0,0,0,0];np.savez(e.parent/'data/raw_data.npz',**a)
    r,c=judge.evaluate(p,e);assert c==2 and not r['metrics']

def test_foreign_cwd_cli_and_report_write_error(tmp_path):
    p=ROOT/'output/reconstruction.npz';report=tmp_path/'nested/report.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert result.returncode==0;assert json.loads(report.read_text())['passed']
    command[-1]=str(tmp_path);result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert result.returncode==2 and 'Report write failed' in result.stderr

@pytest.mark.parametrize('index',range(31))
def test_every_parameter_upper_and_lower_support(bundle,index):
    e,p=bundle
    with np.load(p) as f:original={k:f[k] for k in f.files}
    for boundary,direction in [(judge.BOUNDS[index,0],-np.inf),(judge.BOUNDS[index,1],np.inf)]:
        a={k:v.copy() for k,v in original.items()};a['reconstruction'][0,index]=np.nextafter(boundary,direction)
        np.savez(p,**a);r,c=judge.evaluate(p,e);assert c==2 and r['metrics']=={}

@pytest.mark.parametrize('value',[True,'1',[],{},None,float('nan'),float('inf'),-1])
def test_malformed_configuration_actual_cli_exit_two(bundle,value):
    e,p=bundle;shutil.copy2(ROOT/'evaluation/judge.py',e/'judge.py')
    config=json.loads((e/'metrics.json').read_text());config['weighted_sse_boundary']=value;(e/'metrics.json').write_text(json.dumps(config))
    report=e/'bad_config_report.json'
    result=subprocess.run([sys.executable,str(e/'judge.py'),'--prediction',str(p),'--report',str(report)],cwd=e.parent,capture_output=True,text=True)
    payload=json.loads(report.read_text());assert result.returncode==2 and payload['status']=='error' and payload['metrics']=={}

def test_unknown_boundary_rejected(bundle):
    e,p=bundle;config=json.loads((e/'metrics.json').read_text());config['unused_boundary']=1.;(e/'metrics.json').write_text(json.dumps(config))
    report,code=judge.evaluate(p,e);assert code==2 and report['metrics']=={}

@pytest.mark.parametrize('comparisons',[{},None,[],{'weighted_sse':'<='},{'weighted_sse':'>=','curve_360_relative':'<=','curve_197_relative':'<='},{'weighted_sse':'<=','curve_360_relative':'<=','curve_197_relative':'<=','extra':'<='}])
def test_malformed_comparison_mapping(bundle,comparisons):
    e,p=bundle;config=json.loads((e/'metrics.json').read_text());config['comparisons']=comparisons;(e/'metrics.json').write_text(json.dumps(config))
    report,code=judge.evaluate(p,e);assert code==2 and report['metrics']=={}

def test_complete_upper_bound_comparison_mapping_passes(bundle):
    e,p=bundle;config=json.loads((e/'metrics.json').read_text());config['comparisons']={k:'<=' for k in judge.THRESHOLDS};(e/'metrics.json').write_text(json.dumps(config))
    assert judge.evaluate(p,e)[1]==0

def test_big_endian_float64_submission_is_valid(bundle):
    e,p=bundle
    with np.load(p) as f:arrays={k:f[k].astype('>f8') for k in f.files}
    np.savez(p,**arrays);report,code=judge.evaluate(p,e);assert code==0 and report['passed']
