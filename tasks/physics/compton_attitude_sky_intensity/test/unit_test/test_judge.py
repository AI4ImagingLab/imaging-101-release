from pathlib import Path
import importlib.util,json,subprocess,sys,zipfile
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_compton_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return z['reconstruction']

def prediction(tmp_path,values):
    path=tmp_path/'prediction.npz';np.savez_compressed(path,reconstruction=values);return path

@pytest.mark.parametrize('case',['shape','integer','complex','float16','nan','inf','negative','key','missing'])
def test_invalid_submission(tmp_path,reference,case):
    values=dict(shape=reference.ravel(),integer=reference.astype(np.int64),complex=reference.astype(np.complex64),float16=reference.astype(np.float16),nan=reference.copy(),inf=reference.copy(),negative=-reference)
    if case=='missing':path=tmp_path/'absent.npz'
    elif case=='key':path=tmp_path/'badkey.npz';np.savez_compressed(path,wrong=reference)
    else:
        array=values[case]
        if case=='nan':array[0,0]=np.nan
        if case=='inf':array[0,0]=np.inf
        path=prediction(tmp_path,array)
    report,code=judge.evaluate(path);assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

@pytest.mark.parametrize('value',[True,'0.001',float('nan'),0,-1])
def test_invalid_numeric_configuration(tmp_path,reference,value):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());config['normalized_poisson_deviance_boundary']=value
    directory=tmp_path/'evaluation';directory.mkdir();(directory/'metrics.json').write_text(json.dumps(config))
    report,code=judge.evaluate(prediction(tmp_path,reference),directory,ROOT/'data/raw_data.npz');assert code==2 and report['metrics']=={}

def test_unexpected_truth_and_missing_configuration(tmp_path,reference):
    directory=tmp_path/'evaluation';directory.mkdir();path=prediction(tmp_path,reference)
    assert judge.evaluate(path,directory,ROOT/'data/raw_data.npz')[1]==2
    (directory/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes());(directory/'ground_truth.npz').mkdir()
    report,code=judge.evaluate(path,directory,ROOT/'data/raw_data.npz');assert code==2 and 'Unexpected actual truth' in report['errors'][0]

def test_exact_poisson_scalar_and_zero_cases():
    value,bad=judge.deviance_sum(np.array([0.,1.,3.]),np.array([0.,2.,4.]))
    expected=2*(2-1+np.log(.5)+4-3+3*np.log(.75))
    assert value==pytest.approx(expected) and bad==0
    value,bad=judge.deviance_sum(np.array([0.,1.]),np.array([0.,0.]))
    assert np.isfinite(value) and value>1000 and bad==1

def test_finite_extreme_prediction_has_structured_error_report(tmp_path):
    path=prediction(tmp_path,np.full((1,3072),1e308,dtype=np.float64))
    report,code=judge.evaluate(path);assert code==2 and report['metrics']=={} and report['errors'];json.dumps(report,allow_nan=False)
    destination=tmp_path/'extreme_report.json'
    child=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(destination)],cwd=tmp_path,capture_output=True)
    assert child.returncode==2 and json.loads(destination.read_text())['metrics']=={}

def test_foreign_cwd_and_report_failure(tmp_path,reference):
    path=prediction(tmp_path,reference);destination=tmp_path/'nested/report.json'
    child=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(destination)],cwd=tmp_path,capture_output=True)
    assert child.returncode==0 and json.loads(destination.read_text())['passed']
    blocked=tmp_path/'blocked';blocked.write_text('file blocks directory')
    child=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'missing.npz'),'--report',str(blocked/'report.json')],cwd=tmp_path,capture_output=True)
    assert child.returncode==2 and b'Cannot write judge report' in child.stderr

def test_complete_forward_and_independent_scalar_contractions(tmp_path,reference):
    # The frozen oracle baseline and cleaned stored method output differ at
    # float32 roundoff. Repeat the exact asset used for the attainable bound.
    reference=judge._field(ROOT/'output/reconstruction.npz')[None]
    raw=ROOT/'data/raw_data.npz';context=judge.public_context(raw)
    columns=np.array([0,12345,80000,150000,184319]);attitudes=np.array([0,137,277]);observed={}
    def capture(lo,hi,values):
        for col in columns[(columns>=lo)&(columns<hi)]:observed[int(col)]=values[0,attitudes,col-lo].copy()
    stats=judge.forward_statistics(reference,raw,context,capture)
    assert stats['observation_bins']==51240960 and stats['minimum_available_memory']>=6*1024**3
    config=json.loads((ROOT/'evaluation/metrics.json').read_text())
    assert abs(float(stats['normalized_poisson_deviance'][0])-config['attained_normalized_poisson_deviance'])<1e-12
    selected=np.empty((3072,len(columns)),float)
    with zipfile.ZipFile(raw) as archive,archive.open('response_counts.npy') as handle:
        version=np.lib.format.read_magic(handle);reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=reader(handle);assert dtype==np.dtype('uint16')
        for k in range(3072):selected[k]=np.frombuffer(handle.read(184320*2),dtype=dtype)[columns]
    with np.load(raw,allow_pickle=False) as z:c=z['attitude_coordinates'][0]-1;weights=z['attitude_weights_s'][0].astype(float);area=float(z['response_area_cm2'].item())
    direct=[]
    for attitude in attitudes:
        keep=c[0]==attitude;local=np.bincount(c[2,keep],weights=weights[keep]*reference[0,c[1,keep]],minlength=3072)
        direct.append(local@selected*area*(4*np.pi/3072))
    got=np.array([observed[int(col)] for col in columns]).T
    np.testing.assert_allclose(got,direct,rtol=2e-13,atol=1e-15)
    (tmp_path/'independent_scalar_contractions.json').write_text(json.dumps(dict(max_absolute_difference=float(np.max(abs(got-direct))),full_bins=stats['observation_bins'],minimum_available_host_bytes=stats['minimum_available_memory']),indent=2),encoding='utf8')

def test_unsupported_bins_boundary_is_exact_zero(tmp_path,reference):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());config['unsupported_positive_bins_boundary']=1
    directory=tmp_path/'evaluation';directory.mkdir();(directory/'metrics.json').write_text(json.dumps(config))
    report,code=judge.evaluate(prediction(tmp_path,reference),directory,ROOT/'data/raw_data.npz');assert code==2 and not report['metrics']
