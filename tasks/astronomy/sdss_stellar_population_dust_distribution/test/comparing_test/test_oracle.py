from pathlib import Path
import sys,importlib.util,json,shutil,subprocess
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data,normalize_models
from src.physics_model import highpass,calzetti,forward,adjoint,smooth_curve
from src.solvers import reconstruct,fit_population,sigma_clip
spec=importlib.util.spec_from_file_location('taskjudge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_complete_two_stage_original_parity():
    data=load_data(ROOT/'data/raw_data.npz');out,result=reconstruct(data)
    with np.load(ROOT/'test/fixtures/upstream.npz') as z:
        for name,fit in [('highpass',result['highpass']['fit']),('final',result['final'])]:
            np.testing.assert_allclose(fit['chis'],z[name+'_output_chis'],rtol=1e-9,atol=1e-9)
            np.testing.assert_array_equal(fit['branches'],z[name+'_output_branch'])
            np.testing.assert_allclose(fit['weights'][z[name+'_indices']],z[name+'_output_weights'],rtol=0,atol=1e-8)
        for key in ['flux','error','templates']:np.testing.assert_allclose(result['highpass'][key],z['highpass_input_'+key],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(out['reconstruction'][0],z['final_best_weights'],rtol=0,atol=1e-8)
        np.testing.assert_allclose(out['attenuation'][0],z['output_attenuation'],rtol=1e-10,atol=1e-12)
        np.testing.assert_array_equal(out['ebv'].ravel(),z['output_ebv'])
        expected=z['final_best_weights']@z['final_input_templates'];np.testing.assert_allclose(result['prediction'],expected,rtol=1e-8,atol=1e-12)
    assert not result['highpass']['bad'].any()
    assert result['highpass']['fit']['retained']==217 and result['final']['retained']==242
def test_cli_pipeline():
    run=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True);assert run.returncode==0,run.stderr
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz');assert code==0,report
