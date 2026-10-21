from pathlib import Path
import numpy as np
from src.preprocessing import load_data,pad_flux
from src.physics_model import frequency_grid,psf_transfer,hanning,forward_spectrum
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_independently_corrected_original_stages_and_full_outputs():
    d=load_data(ROOT/'data/raw_data.npz');output,stage=reconstruct(d)
    with np.load(ROOT/'test/fixtures/corrected_stages.npz',allow_pickle=False) as f:
        for key in ['padded_mag','padded_xs','padded_ys','mag_k','psf_k','Kx','Ky','K','H','factor','jx_k','jy_k']:
            np.testing.assert_allclose(stage[key],f[key],rtol=2e-11,atol=2e-10)
        for i,key in enumerate(['jx_full','jy_full']):np.testing.assert_allclose(stage['full_complex'][i],f[key],rtol=2e-11,atol=2e-10)
        assert stage['padded_xs'][1]-stage['padded_xs'][0]==f['dx']
        assert stage['padded_ys'][1]-stage['padded_ys'][0]==f['dy']
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:np.testing.assert_allclose(output['reconstruction'],f['reconstruction'],rtol=2e-11,atol=2e-10)
    assert stage['imaginary_ratio']<1e-12
def test_all_three_cutoff_original_fixtures():
    d=load_data(ROOT/'data/raw_data.npz')
    for cutoff in [.5,1.,1.5]:
        out,_=reconstruct(d,cutoff=cutoff)
        with np.load(ROOT/f'notebooks/reference_outputs/corrected_{cutoff:g}.npz') as f:np.testing.assert_allclose(out['reconstruction'],f['reconstruction'],rtol=2e-11,atol=2e-10)
def test_full_pipeline_writes_output_and_passes_foreign_judge(tmp_path):
    import subprocess,sys,json
    p=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True);assert p.returncode==0,p.stderr
    report=tmp_path/'report.json'
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    r=json.loads(report.read_text());assert r['status']=='pass' and not r['errors']
