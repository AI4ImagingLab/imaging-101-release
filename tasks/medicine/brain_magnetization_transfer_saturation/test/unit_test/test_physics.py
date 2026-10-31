from pathlib import Path
import numpy as np
from src.physics_model import flash_signal,exact_flash_signal,ANGLES,TR
from src.visualization import comparison_metrics
from src.solvers import reconstruct,mtr
from src.preprocessing import analysis_mask,load_data
ROOT=Path(__file__).resolve().parents[2]
def test_independent_bloch_limit():
    a=.001;tr=1e-5;r=.8;A=1000.
    np.testing.assert_allclose(flash_signal(A,r,0,a,tr),exact_flash_signal(A,r,a,tr),rtol=1e-6)
def test_monotonic_saturation_and_scale():
    assert flash_signal(1000,1,.03,.1,.03)<flash_signal(1000,1,0,.1,.03)
    np.testing.assert_allclose(flash_signal(2000,1,.03,.1,.03),2*flash_signal(1000,1,.03,.1,.03))
def test_analytic_inverse_and_common_gain():
    shape=(128,128,96);A=8000.;r=.8;delta=.025
    vals=[flash_signal(A,r,d,a,t) for d,a,t in zip([0,0,delta],ANGLES,TR)]
    data={k:np.full(shape,v) for k,v in zip(('PDw','T1w','MTw'),vals)}
    mask=analysis_mask(data);assert mask.any()
    out=reconstruct(data)
    np.testing.assert_allclose(out['reconstruction'][0][mask],2.5,atol=1e-12)
    np.testing.assert_allclose(out['r1'][0][mask],.8,atol=1e-12)
    scaled=reconstruct({k:v*2 for k,v in data.items()})
    np.testing.assert_allclose(scaled['reconstruction'],out['reconstruction'],atol=1e-12)
    np.testing.assert_allclose(scaled['r1'],out['r1'],atol=1e-12)
def test_metrics_known_case():
    m=np.ones(2,dtype=bool);a=np.array([1.,2.]);b=np.array([2.,4.]);v=comparison_metrics(a,b,m)
    assert np.isclose(v['rmse'],np.sqrt(2.5));assert np.isclose(v['ncc'],1.)
def test_metadata_and_mtr():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    assert '0.028 / 0.018 / 0.028' in text and '6 / 20 / 6' in text and '219351' in text
    np.testing.assert_array_equal(TR,[.028,.018,.028]);np.testing.assert_allclose(ANGLES,np.array([6,20,6])*np.pi/180,rtol=0,atol=1e-16)
    data=load_data(ROOT/'data/raw_data.npz');mask=analysis_mask(data)
    np.testing.assert_allclose(mtr(data,mask)[mask],100*(1-data['MTw'][mask]/data['PDw'][mask]),atol=1e-13)
