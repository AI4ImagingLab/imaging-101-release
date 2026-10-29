"""Independent numerical cases captured by the archived unmodified oracle."""
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import csc_matrix
from src.preprocessing import register_translation,tile_and_correct
from src.physics_model import get_noise_fft
from src.solvers import initialize_components,NMF,positive_lars,update_spatial,update_temporal,constrained_oasisAR1,finetune,release_compatibility_mode

FIX=Path(__file__).resolve().parents[1]/'fixtures'

@pytest.mark.parametrize('frame',[0,1000,2999])
def test_piecewise_registration_and_cubic_warp(frame):
    z=np.load(FIX/f'nonrigid_{frame}.npz',allow_pickle=False)
    corrected,shifts,_,_=tile_and_correct(z['input_frame'],z['input_template'],(50,50),(24,24),(6,6),max_deviation_rigid=3,add_to_movie=296,shifts_opencv=True,border_nan='copy')
    np.testing.assert_allclose(shifts,z['output_shifts'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(corrected,z['output_corrected'],atol=1.2e-4,rtol=1e-6)

def test_spectral_noise_and_psd():
    z=np.load(FIX/'pixel_noise.npz',allow_pickle=False)
    sn,psd=get_noise_fft(z['input_traces'])
    np.testing.assert_allclose(sn,z['output_sn'],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(psd,z['output_psd'],rtol=1e-5,atol=.002)

def test_greedy_rank_one_refinement():
    z=np.load(FIX/'greedy_refinement.npz',allow_pickle=False)
    a,c=finetune(z['input_patch'],z['input_trace'].copy())
    np.testing.assert_allclose(a,z['output_a'],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(c,z['output_c'],rtol=1e-6,atol=1e-6)

def test_patch_background_nmf_state():
    z=np.load(FIX/'background_nmf.npz',allow_pickle=False)
    np.random.set_state(('MT19937',z['param_rng_keys'],int(z['param_rng_pos']),int(z['param_rng_has_gauss']),float(z['param_rng_cached'])))
    model=NMF();W=model.fit_transform(z['input_X'])
    np.testing.assert_allclose(W,z['output_W'],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(model.components_,z['output_H'],rtol=1e-10,atol=1e-10)

def test_selected_one_background_patch():
    z=np.load(FIX/'full_patch_initialization.npz',allow_pickle=False)
    ref=np.load(FIX/'selected_patch0.npz',allow_pickle=False);np.random.seed(0)
    A,C,b,f,_=initialize_components(z['input_Y'].copy(),nb=1)
    r={'A':A,'C':C,'b':b,'f':f}
    from src.solvers import normalize
    normalize(r)
    for k,v in r.items():
        if hasattr(v,'toarray'):v=v.toarray()
        np.testing.assert_allclose(v,ref[k],atol=.01,rtol=3e-4)

def test_quality_and_relative_fluorescence():
    from src.solvers import dff,compute_event_exceptionality
    z=np.load(FIX/'full_quality.npz',allow_pickle=False)
    arrays=np.load(FIX.parents[1]/'notebooks/reference_outputs/upstream_first_all_components.npz',allow_pickle=False)
    selected=z['output_idx'];A=csc_matrix(arrays['A'])[:,selected]
    actual=dff(A,arrays['b'],arrays['C'][selected],arrays['f'],z['input_R'][selected])
    np.testing.assert_allclose(actual,z['output_dff'],rtol=1e-12,atol=1e-12)

def test_independent_original_repeat_is_bitwise_reproducible():
    directory=FIX.parents[1]/'notebooks/reference_outputs'
    for stage in ['motion_outputs','initial_arrays','all_components','final_arrays']:
        first=np.load(directory/f'upstream_first_{stage}.npz',allow_pickle=False)
        repeated=np.load(directory/f'upstream_repeat_{stage}.npz',allow_pickle=False)
        assert first.files==repeated.files
        for key in first.files:np.testing.assert_array_equal(first[key],repeated[key])

def test_identified_fresh_notebook_global_stage():
    directory=FIX.parents[1]/'notebooks/reference_outputs'
    fresh=np.load(directory/'fresh_notebook_review.npz',allow_pickle=False)
    original=np.load(directory/'upstream_first_all_components.npz',allow_pickle=False)
    assert fresh['all_C'].shape==(149,3000)
    np.testing.assert_array_equal(fresh['accepted'],original['idx_components'])
    np.testing.assert_array_equal(fresh['corrected_selected'],fresh['accepted'])
    for measured,reference in [('all_A','A'),('all_C','C')]:
        relative=np.linalg.norm(fresh[measured]-original[reference])/np.linalg.norm(original[reference])
        assert relative<2e-5

@pytest.mark.parametrize('frame',[0,1000,2999])
def test_registration(frame):
    z=np.load(FIX/f'registration_{frame}.npz',allow_pickle=False)
    shift,_,phase=register_translation(z['input_frame'],z['input_template'],upsample_factor=10,max_shifts=(6,6))
    np.testing.assert_allclose(shift,z['output_shift'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(phase,z['output_phase'],atol=1e-12,rtol=0)

def test_full_greedy_hals_initializer():
    z=np.load(FIX/'full_patch_initialization.npz',allow_pickle=False);np.random.seed(42)
    result=initialize_components(z['input_Y'].copy())
    for key,value in zip(['A','C','b','f','centers'],result):
        value=value.toarray() if hasattr(value,'toarray') else value
        np.testing.assert_allclose(value,z['output_'+key],rtol=3e-4,atol=4e-4)

def test_global_background_nmf():
    z=np.load(FIX/'fullpatch_global_nmf.npz',allow_pickle=False)
    model=NMF(2,max_iter=100,tol=1e-10,random_state=1);W=model.fit_transform(z['X'])
    np.testing.assert_allclose(W,z['W'],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(model.components_,z['H'],rtol=1e-10,atol=1e-10)

@pytest.mark.parametrize('fixture',sorted(FIX.glob('positive_lars_*.npz')),ids=lambda x:x.stem)
def test_positive_lars(fixture):
    z=np.load(fixture,allow_pickle=False)
    coef=positive_lars(z['input_X'],z['input_y'],float(z['param_alpha']))
    np.testing.assert_allclose(coef,z['output_coeff'],rtol=1e-9,atol=1e-9)

def test_complete_spatial_update():
    z=np.load(FIX/'full_spatial_update.npz',allow_pickle=False)
    result=update_spatial(z['input_Y'],z['input_A'],z['input_C'].copy(),z['input_b'].copy(),z['input_f'].copy(),z['param_sn'],(31,31))
    for key,value in zip(['A','b','C','f'],result):
        if hasattr(value,'toarray'):value=value.toarray()
        np.testing.assert_allclose(value,z['output_'+key],rtol=1e-9,atol=1e-9)

@pytest.mark.parametrize('ar',[0,1])
def test_complete_temporal_update(ar):
    z=np.load(FIX/'full_spatial_update.npz',allow_pickle=False)
    result=update_temporal(z['input_Y'],csc_matrix(z['output_A']),z['output_b'].copy(),z['output_C'].copy(),z['output_f'].copy(),p=ar)
    ref=np.load(FIX/f'full_temporal_p{ar}.npz',allow_pickle=False)
    for key in ['A','C','b','f','S','YrA']:
        value=result[key];value=value.toarray() if hasattr(value,'toarray') else value
        np.testing.assert_allclose(value,ref['output_'+key],rtol=1e-4,atol=1e-5)

@pytest.mark.parametrize('fixture',sorted(FIX.glob('oasis*.npz')),ids=lambda x:x.stem)
def test_oasis(fixture):
    z=np.load(fixture,allow_pickle=False)
    actual='actual' in fixture.stem
    result=constrained_oasisAR1(z['input_y'],float(z['param_g']),float(z['param_sn']),optimize_b=bool(z['param_optimize_b']),b_nonneg=not actual)
    for key,value in zip(['c','s','b','g','lam'],result):
        np.testing.assert_allclose(value,z['output_'+key],rtol=2e-5,atol=1.5e-3)

def test_release_mode_known_discrepancy_is_explicit():
    from src.solvers import symmetric_half_sample_mode
    triple=np.array([0.,9.,10.])
    assert release_compatibility_mode(triple)==9
    assert symmetric_half_sample_mode(triple)==9.5
    # The released even-length recursion also omits the final candidate interval.
    even=np.array([0.,10.,20.,20.1])
    assert release_compatibility_mode(even)==5
    assert symmetric_half_sample_mode(even)==pytest.approx(20.05)
    z=np.load(FIX/'half_sample_modes.npz',allow_pickle=False)
    actual=np.array([release_compatibility_mode(x) for x in z['input_triples']])
    np.testing.assert_array_equal(actual,z['output_modes'])
