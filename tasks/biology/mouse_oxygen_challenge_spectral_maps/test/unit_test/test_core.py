from pathlib import Path
import importlib.util
import numpy as np
import pytest
from src.preprocessing import load_data
from src.physics_model import forward,oxygenation,spectral_relative_residual
from src.solvers import unmix,reconstruct
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('spectral_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_public_data_contract_and_metadata():
    images,spectra,labels,wavelength,times=load_data();assert images.shape==(76,10,1,111,111) and (labels>0).sum()==1955
    assert np.array_equal(wavelength,[700,730,750,760,770,800,820,840,850,880])
    assert np.linalg.matrix_rank(spectra)==2 and times[0,0]==0 and times[-1,-1]>990
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    with np.load(ROOT/'data/raw_data.npz') as z:
        for k in z.files:assert '| '+k+' |' in readme
    assert 'github' not in readme.lower() and 'fluence' in readme.lower()
    for command in ['epsilon','lambda','sum']:assert chr(92)+command in readme

def test_independently_known_spectral_mixture_units_and_scaling():
    spectra=np.array([[2.,4.,1.,3.],[1.,1.,3.,2.]])
    truth=np.array([[[[[2.,3.]]],[[[6.,1.]]]]],dtype=np.float32)
    image=forward(truth,spectra);a=unmix(image,spectra)
    assert np.array_equal(a,truth);assert np.array_equal(oxygenation(a),np.array([[[[.75,.25]]]],np.float32))
    scaled=unmix(image*2,spectra);assert np.array_equal(scaled,truth*2)
    assert np.array_equal(oxygenation(scaled),oxygenation(a))
    assert spectral_relative_residual(a,image,spectra,np.ones((1,1,2),bool))==0

def test_zero_total_and_unconstrained_negative_conventions():
    a=np.array([[[[[1.,-1.,0.]]],[[[3.,2.,0.]]]]],np.float32)
    out=oxygenation(a);assert out[0,0,0,0]==.75 and out[0,0,0,1]==2 and np.isnan(out[0,0,0,2])
    image=forward(a,np.array([[1.,2.,1.],[3.,1.,2.]]));assert np.allclose(unmix(image,np.array([[1.,2.,1.],[3.,1.,2.]])),a,atol=1e-14)
    with pytest.raises(ValueError):reconstruct(np.zeros((1,3,1,1,3)),np.array([[1.,2.,1.],[3.,1.,2.]]),np.ones((1,1,3),np.uint8))

def test_normal_equation_orthogonality():
    raw,spectra,labels,_,_=load_data();a=unmix(raw,spectra).astype(np.float64);residual=forward(a,spectra)-raw
    gradient=np.einsum('tw...,cw->tc...',residual,spectra)
    assert np.max(abs(gradient[...,labels>0]))/np.max(abs(np.einsum('tw...,cw->tc...',raw,spectra)))<2e-7

def test_known_metrics_and_degenerate_reference():
    m=judge.metrics(np.array([1.,3.]),np.array([2.,4.]));assert m['ncc']==pytest.approx(14/np.sqrt(200));assert m['nrmse']==.5
    assert judge.metrics(np.zeros(2),np.array([2.,4.]))['ncc']==0
    with pytest.raises(ValueError):judge.metrics(np.ones(2),np.ones(2))

@pytest.mark.parametrize('kind',['missing','key','extra','shape','dtype','complex','nan','inf','object','broken','exterior'])
def test_malformed_submissions(tmp_path,kind):
    p=tmp_path/'bad.npz';a={k:np.zeros(s,np.float32) for k,s in judge.SHAPES.items()}
    if kind=='missing':pass
    elif kind=='key':np.savez(p,reconstruction=a['reconstruction'])
    elif kind=='extra':np.savez(p,**a,extra=[1])
    elif kind=='broken':p.write_bytes(b'broken')
    else:
        if kind=='shape':a['saturation']=a['saturation'][0]
        elif kind=='dtype':a['saturation']=a['saturation'].astype(np.float64)
        elif kind=='complex':a['saturation']=a['saturation'].astype(np.complex64)
        elif kind=='object':a['saturation']=np.array([{}],object)
        else:a['saturation'][0,0,0,0,0]=np.nan if kind=='nan' else np.inf if kind=='inf' else 1e-12
        np.savez(p,**a)
    r=judge.evaluate(p);assert r['status']=='error' and not r['passed'] and r['metrics']=={} and r['errors']

def test_poor_and_missing_evaluator_assets(tmp_path):
    p=tmp_path/'poor.npz';np.savez(p,**{k:np.zeros(s,np.float32) for k,s in judge.SHAPES.items()})
    assert judge.evaluate(p)['status']=='fail'
    assert judge.evaluate(p,tmp_path)['status']=='error'
    (tmp_path/'metrics.json').write_text('{"ncc_boundary":NaN}')
    assert judge.evaluate(p,tmp_path)['status']=='error'

@pytest.mark.parametrize('field',['Hb','HbO2','saturation'])
def test_all_physical_fields_are_scored(tmp_path,field):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:a={k:z[k] for k in z.files}
    if field=='saturation':a['saturation']*=0
    else:a['reconstruction'][:,:,0 if field=='Hb' else 1]*=0
    p=tmp_path/'poor.npz';np.savez(p,**a);r=judge.evaluate(p);assert r['status']=='fail' and r['metrics'][field+'_ncc']==0

def test_cli_foreign_cwd_exitcodes_and_report_error(tmp_path):
    import subprocess,sys,json
    def call(p,r):return subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(r)],cwd=tmp_path,capture_output=True)
    report=tmp_path/'nested/report.json';assert call(ROOT/'evaluation/baseline_reference.npz',report).returncode==0
    assert json.loads(report.read_text())['passed'];assert call(tmp_path/'absent.npz',report).returncode==2
    p=tmp_path/'poor.npz';np.savez(p,**{k:np.zeros(s,np.float32) for k,s in judge.SHAPES.items()});assert call(p,report).returncode==1
    blocker=tmp_path/'file';blocker.write_text('x');assert call(p,blocker/'report.json').returncode==2

def test_main_import_preserves_backend():
    import matplotlib
    before=matplotlib.get_backend();import src.main
    assert matplotlib.get_backend()==before

def test_known_enhancement_window_summary():
    from src.physics_model import window_summary
    signal=np.full((30,1,2,2),.2,np.float32);signal[15:]=.5
    mean,std,delta=window_summary(signal,[0,10,25],buffer_width=5)
    assert np.allclose(mean,.2) and np.allclose(std,0) and np.allclose(delta,.3)
