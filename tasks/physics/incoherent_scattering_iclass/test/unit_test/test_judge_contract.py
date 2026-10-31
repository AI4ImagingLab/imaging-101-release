"""Standalone Judge numerical definitions, malformed inputs and failure exits."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def submission():
    with np.load(ROOT/'notebooks/reference_outputs/upstream_paper_corrected.npz',allow_pickle=False) as archive:
        return {key:archive[key] for key in judge.KEYS}


def test_independent_metric_known_case():
    reference=np.array([[0.,1.],[2.,3.]])
    ncc,nrmse=judge.aligned_metrics(reference,reference)
    assert ncc==pytest.approx(1)
    assert nrmse==0
    ncc,nrmse=judge.aligned_metrics(np.zeros_like(reference),reference)
    assert ncc==0 and nrmse==pytest.approx(np.sqrt(3.5)/3)


@pytest.mark.parametrize('case',['missing_key','shape','dtype','nan','inf','complex_image','phase_magnitude','negative'])
def test_malformed_arrays_have_empty_metrics(tmp_path,case):
    prediction=submission()
    if case=='missing_key':del prediction['mtf']
    if case=='shape':prediction['reconstruction']=prediction['reconstruction'][0]
    if case=='dtype':prediction['reconstruction']=prediction['reconstruction'].astype(np.float64)
    if case=='nan':prediction['reconstruction'][0,0,0]=np.nan
    if case=='inf':prediction['phase'][0,0,0]=np.inf
    if case=='complex_image':prediction['reconstruction']=prediction['reconstruction'].astype(np.complex64)
    if case=='phase_magnitude':prediction['phase']*=2
    if case=='negative':prediction['mtf'][0,0,0]=-1
    path=tmp_path/'bad.npz';np.savez(path,**prediction)
    report=judge.evaluate(path)
    assert report['status']=='error' and not report['passed']
    assert report['metrics']=={} and report['errors']


def test_missing_file_and_hidden_assets(tmp_path):
    assert judge.evaluate(tmp_path/'absent.npz')['status']=='error'
    prediction=tmp_path/'prediction.npz';np.savez(prediction,**submission())
    assert judge.evaluate(prediction,evaluation_dir=tmp_path)['status']=='error'


def test_good_prediction_and_zero_quality(tmp_path):
    prediction=submission();path=tmp_path/'good.npz';np.savez(path,**prediction)
    assert judge.evaluate(path)['passed']
    prediction['reconstruction'].fill(1)
    path=tmp_path/'zero.npz';np.savez(path,**prediction)
    report=judge.evaluate(path)
    assert report['status']=='fail' and not report['passed']
    assert report['metrics']['image_consistency']>0.9


def test_cli_missing_file_exit_two_and_report_path(tmp_path):
    report_path=tmp_path/'nested/report.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'none.npz'),'--report',str(report_path)]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert result.returncode==2
    report=json.loads(report_path.read_text())
    assert report['status']=='error' and report['metrics']=={}


def test_report_write_failure_exit_two(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'none.npz'),'--report',str(tmp_path)]
    result=subprocess.run(command,capture_output=True)
    assert result.returncode==2 and b'Cannot write' in result.stderr


@pytest.mark.parametrize('good',[True,False])
def test_cli_pass_fail_from_other_directory(tmp_path,good):
    prediction=submission()
    if not good:prediction['reconstruction'].fill(1)
    path=tmp_path/'prediction.npz';np.savez(path,**prediction)
    report_path=tmp_path/'report.json'
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),
                           '--prediction',str(path),'--report',str(report_path)],
                          cwd=tmp_path,capture_output=True)
    assert result.returncode==(0 if good else 1)
    assert json.loads(report_path.read_text())['status']==('pass' if good else 'fail')


def test_invalid_metric_definition(tmp_path):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text())
    config['thresholds']['image_consistency']['comparison']='=='
    (tmp_path/'metrics.json').write_text(json.dumps(config))
    path=tmp_path/'prediction.npz';np.savez(path,**submission())
    report=judge.evaluate(path,evaluation_dir=tmp_path)
    assert report['status']=='error' and report['metrics']=={}


@pytest.mark.parametrize('good',[True,False])
def test_observation_defined_phase_candidates(good,tmp_path):
    from scipy import fft
    prediction=submission()
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:x=z['camera_frames'][0].astype(float)
    hann=np.hanning(400);x*=hann[None,:,None]*hann[None,None,:];x-=x.mean(axis=0)
    spectrum=fft.fftshift(fft.fft2(x))
    if good:
        phase=prediction['phase'][0]*np.exp(.05j*np.cos(np.arange(400)[None,:]*2*np.pi/400))
    else:phase=np.exp(1j*np.random.default_rng(42).uniform(-np.pi,np.pi,(400,400)))
    prediction['phase']=phase[None].astype(np.complex64)
    reconstructed=fft.ifft2(fft.ifftshift(spectrum*prediction['phase']/(prediction['mtf']+.01),axes=(-2,-1)))
    image=np.sqrt(np.mean(np.abs(reconstructed)**2,axis=0));image/=image.max()
    prediction['reconstruction']=image[None].astype(np.float32)
    path=tmp_path/'distinct.npz';np.savez(path,**prediction)
    report=judge.evaluate(path)
    assert report['status']==('pass' if good else 'fail'),report
    assert report['metrics']['image_consistency']<.0002
    if not good:assert report['metrics']['covariance_coherence']<6.906686629445275
