from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_data,CAMERAS,observations
from src.solvers import reconstruct_camera,reconstruct

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def data():return load_data()

@pytest.mark.parametrize('camera',CAMERAS)
def test_original_stages_and_final(data,camera):
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        d=reconstruct_camera(data,camera)
        np.testing.assert_allclose(d['direct'],z[camera+'_direct'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(d['dispersed'],z[camera+'_dispersed'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(d['weights'],z[camera+'_pix2wave_map_weights'],rtol=0,atol=1e-8)
        np.testing.assert_allclose(d['system']['matrix'],z[camera+'_A_lag'],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(d['system']['rhs'],z[camera+'_b_lag'],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(d['system']['products'],z[camera+'_vec_a'],rtol=1e-8,atol=1e-10)
        ref=z[camera+'_sensitivity'];ref=ref/ref.max()
        np.testing.assert_allclose(d['reconstruction'],ref,rtol=0,atol=1e-7)
        np.testing.assert_allclose(d['pred_direct'],d['direct'],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(d['system']['matrix']@d['solution'],d['system']['rhs'],rtol=1e-8,atol=1e-10)

@pytest.mark.parametrize('variant,settings',[('led_only',{'fluorescent':False}),('sensitivity5',{'basis_size':5}),('sensitivity9',{'basis_size':9}),('attenuation3',{'attenuation_size':3}),('attenuation5',{'attenuation_size':5})])
def test_original_scientific_comparisons(data,variant,settings):
    with np.load(ROOT/'test/fixtures/upstream_comparisons.npz',allow_pickle=False) as z:
        for c in CAMERAS:
            d=reconstruct_camera(data,c,**settings);key=c+'_'+variant+'__final__'
            ref=z[key+'sensitivity'];ref=ref/ref.max()
            np.testing.assert_allclose(d['reconstruction'],ref,rtol=0,atol=1e-7)
            np.testing.assert_allclose(d['weights'],z[key+'pix2wave_map_weights'],rtol=0,atol=1e-8)

def test_cli_and_ground_truth_judge(tmp_path):
    import subprocess,sys,json
    subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,check=True,capture_output=True)
    report=tmp_path/'judge.json'
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert p.returncode==0,p.stderr
    r=json.loads(report.read_text());assert r['passed'] and len(r['metrics'])==9

def test_target_calibration_excluded_and_dataset_shapes(data):
    assert data['population_sensitivity'].shape==(44,31,3)
    assert len(data)==17
    for c in CAMERAS:
        assert data[c+'_direct'].dtype==np.float32
        assert data[c+'_diffracted'].shape==data[c+'_fluorescent'].shape
    assert set(data).isdisjoint({'ground_truth','reference','sensitivity','reconstruction'})
    np.testing.assert_allclose(data['population_wavelength'],np.arange(400,701,10)*1e-9,atol=1e-20)
    np.testing.assert_allclose(data['illumination_wavelength'],np.arange(400,721)*1e-9,atol=1e-20)
