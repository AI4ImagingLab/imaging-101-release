from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import fit_voxel
from src.physics_model import signal
ROOT=Path(__file__).resolve().parents[2]
RTOL=1e-6
ATOL=1e-8

def test_original_deterministic_stages_and_stochastic_fit_contracts():
    im,mask,config_bvalues,_=load_data(ROOT/'data/raw_data.npz');coords=np.argwhere(mask)
    for method in ['one_step','two_step']:
        for i in [0,1261,2522]:
            input_observation=im[tuple(coords[i])]
            # Different RNG stream is only a reproducible diagnostic setting;
            # no random-search final point is matched to the upstream seed.
            fitted,stages=fit_voxel(input_observation,config_bvalues,20261029+i,method,True)
            with np.load(ROOT/f'test/fixtures/{method}_stage_{i}.npz',allow_pickle=False) as z:
                # Stage0 uses deterministic polyfit/CG, with no RNG consumption.
                for key in ['params','prediction','residual','chisqr']:
                    np.testing.assert_allclose(stages[0][key],z[f'stage_0_{key}'],rtol=RTOL,atol=ATOL)
                # Historical DE estimates are fixed *inputs* to the deterministic
                # acquisition/objective checks, not exact random-search answers.
                for j in [1,2]:
                    input_parameters=z[f'stage_{j}_params'].copy()
                    output_prediction=z[f'stage_{j}_prediction']
                    output_residual=z[f'stage_{j}_residual']
                    output_chisqr=z[f'stage_{j}_chisqr']
                    computed=signal(config_bvalues,*input_parameters)
                    residual=input_observation-computed
                    np.testing.assert_allclose(computed,output_prediction,rtol=RTOL,atol=ATOL)
                    np.testing.assert_allclose(residual,output_residual,rtol=RTOL,atol=ATOL)
                    np.testing.assert_allclose(np.sum(residual**2),output_chisqr,rtol=RTOL,atol=ATOL)
            assert fitted.shape==(4,) and fitted.dtype==np.float64 and np.isfinite(fitted).all()
            assert len(stages)==3
            D=stages[0]['params'][1];base=np.exp(stages[0]['params'][0])
            config_lower=np.array([.7*base,.00015,0.,2*D])
            config_upper=np.array([1.7*base,1.2*D,.2,.05])
            if method=='two_step':
                config_lower[3]=D;config_upper[3]=.0355;config_upper[2]=.35
            for j in [1,2]:
                fresh=stages[j];parameters=fresh['params']
                assert parameters.shape==(4,) and parameters.dtype==np.float64
                assert np.isfinite(parameters).all()
                assert ((parameters>=config_lower)&(parameters<=config_upper)).all()
                if method=='two_step':assert parameters[1]==D  # Deterministically fixed model coordinate.
                computed=signal(config_bvalues,*parameters)
                residual=input_observation-computed
                assert fresh['prediction'].shape==config_bvalues.shape and np.isfinite(fresh['prediction']).all()
                assert fresh['residual'].shape==config_bvalues.shape and np.isfinite(fresh['residual']).all()
                np.testing.assert_allclose(fresh['prediction'],computed,rtol=RTOL,atol=ATOL)
                np.testing.assert_allclose(fresh['residual'],residual,rtol=RTOL,atol=ATOL)
                np.testing.assert_allclose(fresh['chisqr'],np.sum(residual**2),rtol=RTOL,atol=ATOL)
                # Source's second search box is defined by its own first estimate.
                config_lower=.95*parameters;config_upper=1.05*parameters
                if parameters[2]<=1e-12:config_lower[2]=0.;config_upper[2]=.05

def test_full_pipeline(tmp_path):
    from src.main import run
    from evaluation.judge import evaluate
    target=tmp_path/'prediction.npz';run(ROOT/'data/raw_data.npz',target)
    report,code=evaluate(target);assert code==0,report
