from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_fractional_shift_and_inversion_have_no_fitted_gain():
    shape=(12,10,14);rng=np.random.default_rng(307)
    F=np.zeros(shape,dtype=complex)
    for h in [(1,2,1),(2,1,-2),(1,-1,3),(3,2,-1)]:
        value=rng.normal()+1j*rng.normal();F[h]=value
        F[tuple((-np.array(h))%shape)]=value.conjugate()
    reference=np.fft.ifftn(F).real;shift=np.array([.31,-.27,.43])
    phase=np.exp(-2j*np.pi*sum(np.fft.fftfreq(n).reshape(tuple(n if k==i else 1 for k in range(3)))*shift[i] for i,n in enumerate(shape)))
    moved=np.fft.ifftn(F*phase).real
    inverted=moved[np.ix_(*[(-np.arange(n))%n for n in shape])]
    for candidate in [moved,inverted]:
        aligned,_,_=judge.align_periodic(candidate,reference)
        np.testing.assert_allclose(aligned,reference,atol=2e-8)
    aligned,_,_=judge.align_periodic(2*moved,reference)
    np.testing.assert_allclose(np.linalg.norm(aligned),2*np.linalg.norm(reference),rtol=1e-12)

@pytest.mark.parametrize('case',['missing','key','shape','integer','complex','nan','inf','object'])
def test_invalid_prediction_empty_metrics(case,tmp_path):
    path=tmp_path/'prediction.npz';x=np.zeros(judge.SHAPE)
    if case=='key':np.savez(path,other=x)
    elif case=='shape':np.savez(path,reconstruction=x[0])
    elif case=='integer':np.savez(path,reconstruction=x.astype(np.int16))
    elif case=='complex':np.savez(path,reconstruction=x.astype(complex))
    elif case=='object':np.savez(path,reconstruction=np.array([object()],dtype=object))
    elif case in ('nan','inf'):
        x.flat[0]=np.nan if case=='nan' else np.inf;np.savez(path,reconstruction=x)
    report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_zero_prediction_is_valid_but_fails(tmp_path):
    path=tmp_path/'zero.npz';np.savez(path,reconstruction=np.zeros(judge.SHAPE))
    report,code=judge.evaluate(path)
    assert code==1 and report['status']=='fail' and not report['passed']
    assert report['metrics']['ncc']==0 and report['metrics']['intensity_relative_l2']==1

@pytest.mark.parametrize('kind',['scale','wrong_sign','shuffle'])
def test_poor_full_maps_fail_without_dropping_observations(kind,tmp_path):
    x=np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False)['reconstruction'].copy()
    if kind=='scale':x*=2
    elif kind=='wrong_sign':x*=-1
    else:x=np.random.default_rng(841).permutation(x.ravel()).reshape(x.shape)
    path=tmp_path/(kind+'.npz');np.savez(path,reconstruction=x)
    report,code=judge.evaluate(path)
    assert code==1 and not report['passed'] and set(report['metrics'])=={'ncc','nrmse','intensity_relative_l2','mean_relative'}

def test_float32_contract_preserves_values(tmp_path):
    x=np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False)['reconstruction'].astype(np.float32)
    path=tmp_path/'float32.npz';np.savez(path,reconstruction=x)
    np.testing.assert_array_equal(judge.load_density(path,'prediction'),x[0])

@pytest.mark.parametrize('boundary',[True,float('nan'),-1])
def test_invalid_thresholds_error_before_scoring(boundary,tmp_path):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());config['ncc_boundary']=boundary
    (tmp_path/'metrics.json').write_text(json.dumps(config))
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and report['metrics']=={} and report['errors']

def test_truth_takes_precedence_and_invalid_truth_never_falls_back(tmp_path):
    assets=tmp_path/'evaluation';assets.mkdir()
    (assets/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes())
    (assets/'baseline_reference.npz').write_bytes((ROOT/'evaluation/baseline_reference.npz').read_bytes())
    np.savez(assets/'ground_truth.npz',reconstruction=np.zeros(judge.SHAPE))
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',assets)
    assert code==2 and 'Degenerate evaluator reference' in report['errors'][0]

def test_missing_evaluator_assets_are_error(tmp_path):
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and report['metrics']=={}

def test_cli_missing_prediction_nested_report_and_write_failure(tmp_path):
    report=tmp_path/'nested/report.json'
    args=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'missing.npz'),'--report',str(report)]
    result=subprocess.run(args,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2 and json.loads(report.read_text())['status']=='error'
    blocked=tmp_path/'blocked';blocked.write_text('file')
    args[-1]=str(blocked/'report.json')
    result=subprocess.run(args,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2 and 'Cannot write evaluation report' in result.stderr
