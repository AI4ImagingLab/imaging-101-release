from pathlib import Path
import sys,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data
from src.physics_model import predictions,depth_profiles,layer_stack
from src.solvers import fit_model,LOWER,UPPER
spec=importlib.util.spec_from_file_location('judge_oracle',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_both_native_forward_layers_profiles_and_inverses():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as z:
        for name in ['uniform','dead']:
            dead=name=='dead';p=z['output_'+name+'_parameters'];expected=np.stack([z['output_'+name+'_'+s+'_prediction'] for s in ['pp','mm']])
            np.testing.assert_allclose(predictions(p,data,dead),expected,rtol=1e-8,atol=1e-12)
            np.testing.assert_allclose(judge.predictions(p,data,dead),expected,rtol=1e-8,atol=1e-12)
            profile=z['output_'+name+'_smooth_profile'];local=depth_profiles(p,profile[0],dead)
            np.testing.assert_allclose(local,np.stack([profile[1],profile[3]],axis=-1),rtol=1e-10,atol=1e-12)
            width,rho,mag,sigma=layer_stack(p,dead)
            np.testing.assert_array_equal(rho,z['output_'+name+'_slab_rho'][0][::-1]);np.testing.assert_array_equal(mag,z['output_'+name+'_slab_rhoM'][::-1]);np.testing.assert_array_equal(sigma,z['output_'+name+'_slab_sigma'][::-1])
            np.testing.assert_allclose(width[1:-1],z['output_'+name+'_slab_w'][::-1][1:-1],rtol=1e-14)
            fit=fit_model(data,dead);keep=np.arange(9) if dead else np.array([0,1,2,3,4,6,7,8])
            assert np.max(abs(fit['parameters']-p)/(UPPER-LOWER)[keep])<=1e-3
            expected_obj=.5*np.sum(z['output_'+name+'_residual']**2)
            assert abs(fit['objective']-expected_obj)<=1e-4
            assert fit['scaled_size']<=1e-7 and fit['objective_spread']<=1e-9

def test_full_cli():
    import subprocess
    result=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True);assert result.returncode==0,result.stderr
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz');assert code==0,report
