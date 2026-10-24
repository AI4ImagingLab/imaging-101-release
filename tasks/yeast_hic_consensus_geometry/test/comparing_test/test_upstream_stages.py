from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_data, ice_normalize, build_pairs
from src.physics_model import wish_distances, poisson_objective, poisson_gradient, mds_objective, mds_gradient

ROOT=Path(__file__).resolve().parents[2]

def test_native_observation_identity_and_support():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/original_stages_prefit.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(data['contact_counts'][0],f['counts'])
        np.testing.assert_array_equal(data['bead_support'][0],f['mask'])

def test_independent_original_ice_and_power_law_stages():
    with np.load(ROOT/'test/fixtures/original_stages_prefit.npz',allow_pickle=False) as f:
        row,col,counts=build_pairs(f['counts'])
        normalized,bias,history=ice_normalize(f['counts'])
        assert_allclose(normalized,f['normalized'],rtol=1e-10,atol=1e-10)
        assert_allclose(bias,f['bias'].reshape(-1),rtol=1e-10,atol=1e-10)
        assert_allclose(wish_distances(row,col,counts,bias),f['wish'][row,col],rtol=1e-10,atol=1e-10)

def test_independent_original_poisson_loss_and_gradient():
    with np.load(ROOT/'test/fixtures/original_stages_prefit.npz',allow_pickle=False) as f:
        row,col,c=build_pairs(f['counts']);b=f['bias'].reshape(-1);x=f['initial']
        assert_allclose(poisson_objective(x,row,col,c,b),f['loss0'],rtol=1e-10,atol=1e-10)
        assert_allclose(poisson_gradient(x,row,col,c,b),f['gradient0'],rtol=1e-10,atol=1e-10)

def test_independent_original_mds_stage():
    with np.load(ROOT/'test/fixtures/original_stages_prefit.npz',allow_pickle=False) as f,np.load(ROOT/'test/fixtures/original_mds_stage.npz',allow_pickle=False) as g:
        r,c,n=build_pairs(f['counts']);wish=wish_distances(r,c,n,f['bias'].reshape(-1));x=g['coordinates']
        assert_allclose(mds_objective(x,r,c,wish),g['loss'],rtol=1e-10,atol=1e-10)
        assert_allclose(mds_gradient(x,r,c,wish).ravel(),g['gradient'],rtol=1e-10,atol=1e-10)

def test_analytic_poisson_and_independent_finite_difference():
    with np.load(ROOT/'test/fixtures/independent_cases.npz',allow_pickle=False) as f:
        args=(f['coordinates'],f['row'],f['col'],f['counts'],f['bias'])
        assert_allclose(poisson_objective(*args),f['original_objective'],rtol=0,atol=1e-12)
        assert_allclose(poisson_gradient(*args),f['finite_difference_gradient'],rtol=0,atol=1e-7)

def test_original_isotonic_and_profiled_alpha_stages():
    from src.solvers import monotone_fit
    from src.physics_model import alpha_profile
    with np.load(ROOT/'test/fixtures/original_isotonic_stage.npz',allow_pickle=False) as f:
        assert_allclose(monotone_fit(f['inverse_count'],f['observed_distance']),f['isotonic_distance'],rtol=1e-10,atol=1e-10)
    with np.load(ROOT/'test/fixtures/original_stages_prefit.npz',allow_pickle=False) as f,np.load(ROOT/'test/fixtures/original_mds_stage.npz',allow_pickle=False) as g,np.load(ROOT/'test/fixtures/original_alpha_profile.npz',allow_pickle=False) as h:
        r,c,n=build_pairs(f['counts'])
        for k,alpha in enumerate(h['alpha']):
            value,gradient,beta=alpha_profile(g['coordinates'],r,c,n,f['bias'].ravel(),alpha)
            assert_allclose([value,gradient,beta],[h['objective'][k],h['gradient'][k],h['beta'][k]],rtol=1e-10,atol=1e-8)
