from pathlib import Path
import importlib.util,json,subprocess,sys,shutil,numpy as np,pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_stellar_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def valid():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in ['reconstruction','spectrum']}

def test_independent_parameter_metric_known_case(tmp_path):
    a=valid()
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:scale=z['parameter_scale']
    a['reconstruction']=a['reconstruction']+2*scale;p=tmp_path/'prediction.npz';np.savez(p,**a)
    report=judge.evaluate(p);np.testing.assert_allclose(report['metrics']['parameter_standardized_max'],2.,rtol=1e-12,atol=1e-12)
    assert report['passed']
    a['reconstruction']+=10*scale;np.savez(p,**a);assert judge.evaluate(p)['status']=='fail'

def test_complete_spectrum_metric_known_case(tmp_path):
    a=valid();dynamic=np.ptp(a['spectrum']);a['spectrum']+=.002*dynamic
    p=tmp_path/'prediction.npz';np.savez(p,**a);report=judge.evaluate(p)
    np.testing.assert_allclose(report['metrics']['spectrum_nrmse'],.002,rtol=1e-12,atol=1e-12)
    assert report['status']=='fail'

@pytest.mark.parametrize('kind',['missing','key','parameter_shape','spectrum_shape','float32','complex','nan','inf'])
def test_invalid_inputs(kind,tmp_path):
    p=tmp_path/'bad.npz';a=valid()
    if kind=='key':a.pop('spectrum')
    elif kind=='parameter_shape':a['reconstruction']=a['reconstruction'].T
    elif kind=='spectrum_shape':a['spectrum']=a['spectrum'][0]
    elif kind=='float32':a['spectrum']=a['spectrum'].astype('float32')
    elif kind=='complex':a['reconstruction']=a['reconstruction'].astype('complex128')
    elif kind in ['nan','inf']:a['spectrum'][0,0]=np.nan if kind=='nan' else np.inf
    if kind!='missing':np.savez(p,**a)
    report=judge.evaluate(p);assert report['status']=='error' and report['metrics']=={} and report['errors']

def test_assets_precedence_and_invalid_configuration(tmp_path):
    # Public observations needed by the independent consistency score are copied
    # as only small arrays; no solver or model-calibration imports occur.
    assets=tmp_path/'evaluation';assets.mkdir();data=tmp_path/'data';data.mkdir()
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:np.savez(data/'raw_data.npz',**{k:z[k] for k in ['wavelength','flux','masks']})
    shutil.copy2(ROOT/'evaluation/metrics.json',assets/'metrics.json')
    p=tmp_path/'prediction.npz';a=valid();np.savez(p,**a)
    assert judge.evaluate(p,assets)['status']=='error'
    shutil.copy2(ROOT/'evaluation/baseline_reference.npz',assets/'baseline_reference.npz')
    truth={k:v.copy() for k,v in a.items()};truth['reconstruction']+=.001;np.savez(assets/'ground_truth.npz',**truth);np.savez(p,**truth)
    assert judge.evaluate(p,assets)['metrics']['parameter_standardized_max']==0.
    truth['spectrum'][:]=1.;np.savez(assets/'ground_truth.npz',**truth);assert judge.evaluate(p,assets)['status']=='error'
    config=json.loads((assets/'metrics.json').read_text());config['spectrum_nrmse_boundary']=0.;(assets/'metrics.json').write_text(json.dumps(config));assert judge.evaluate(p,assets)['status']=='error'

def test_cli_codes_paths_and_write_error(tmp_path):
    p=tmp_path/'prediction.npz';a=valid();np.savez(p,**a);report=tmp_path/'created'/'report.json'
    args=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)]
    result=subprocess.run(args,cwd=tmp_path,capture_output=True);assert result.returncode==0 and json.loads(report.read_text())['passed']
    a['reconstruction'][:]=0;np.savez(p,**a);assert subprocess.run(args,capture_output=True).returncode==1
    a.pop('spectrum');np.savez(p,**a);assert subprocess.run(args,capture_output=True).returncode==2
    result=subprocess.run(args[:-1]+[str(tmp_path)],capture_output=True);assert result.returncode==2 and result.stderr
