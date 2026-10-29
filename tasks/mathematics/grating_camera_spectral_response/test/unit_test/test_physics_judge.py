from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
from src.physics_model import interpolation_matrix,fourier_basis,wavelength_kernel,predicted_observations
from src.solvers import peak_mapping

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_interpolation_affine_extrapolation():
    query=np.array([-2.,0,.5,2.,4]);grid=np.arange(4.)
    m=interpolation_matrix(query,grid)
    np.testing.assert_allclose(m@(2*grid+3),2*query+3)
    np.testing.assert_allclose(m.sum(axis=1),1)

def test_wavelength_support_and_affine_weights():
    x=np.array([-.1,0.,.5,2.,3.,3.4,3.6]);k=wavelength_kernel(x,4)
    np.testing.assert_allclose(k.sum(axis=1),[0,1,1,1,1,1,0])
    np.testing.assert_allclose(k@np.arange(4.),[0,0,.5,2,3,3,0])
    assert np.min(k)>=0

def test_fourier_and_forward():
    b=fourier_basis(9,5)
    np.testing.assert_array_equal(b[:,0],1)
    np.testing.assert_allclose(b[0],b[-1],atol=1e-14)
    s=np.arange(12,dtype=float).reshape(4,3)+1;e=np.ones(4);eta=np.full(4,.2)
    direct,dispersed=predicted_observations(s,eta,e,np.eye(4),np.eye(4))
    np.testing.assert_array_equal(direct,s.sum(axis=0));np.testing.assert_allclose(dispersed,s*.2)
    with pytest.raises(ValueError):fourier_basis(9,4)

def test_peak_mapping_known_line():
    a=np.zeros((12,3));b=np.zeros((25,3))
    for c in range(3):
        a[2+2*c,c]=2;a[3+2*c,c]=1;b[2*(2+2*c)+1,c]=2;b[2*(3+2*c)+1,c]=1
    np.testing.assert_allclose(peak_mapping(a,b,degree=2),[2,1],atol=1e-12)

def test_metric_known_case_and_zero():
    a=np.array([0.,2,2]);r=np.array([0.,1,2])
    ncc,nrmse=judge.pair_metrics(a,r)
    assert ncc==pytest.approx(6/np.sqrt(40));assert nrmse==pytest.approx(1/np.sqrt(3)/2)
    assert judge.pair_metrics(a*0,r)[0]==0
    with pytest.raises(ValueError):judge.pair_metrics(a,np.ones(3))

@pytest.mark.parametrize('case',['missing','key','shape','integer','complex','nan','inf','pickle','broken'])
def test_judge_malformed_prediction(tmp_path,case):
    path=tmp_path/'submission.npz';a=np.ones(judge.SHAPE)
    if case=='missing':pass
    elif case=='key':np.savez(path,wrong=a)
    elif case=='shape':np.savez(path,reconstruction=a[:,0])
    elif case=='integer':np.savez(path,reconstruction=a.astype('int32'))
    elif case=='complex':np.savez(path,reconstruction=a.astype('complex128'))
    elif case in ['nan','inf']:
        a.ravel()[0]=float(case);np.savez(path,reconstruction=a)
    elif case=='pickle':np.savez(path,reconstruction=a.astype(object))
    elif case=='broken':path.write_bytes(b'not an archive')
    r=judge.evaluate(path);assert r['status']=='error' and r['metrics']=={} and r['errors']

def test_judge_poor_output_and_missing_assets(tmp_path):
    path=tmp_path/'bad.npz';np.savez(path,reconstruction=np.zeros(judge.SHAPE))
    r=judge.evaluate(path);assert r['status']=='fail' and not r['passed']
    r=judge.evaluate(path,assets=tmp_path);assert r['status']=='error' and not r['metrics']

def test_judge_report_write_failure_and_exit(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'none.npz'),'--report',str(tmp_path/'report.json')],capture_output=True)
    assert p.returncode==2 and json.loads((tmp_path/'report.json').read_text())['status']=='error'
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path)],capture_output=True)
    assert p.returncode==2 and b'Report-write failure' in p.stderr
