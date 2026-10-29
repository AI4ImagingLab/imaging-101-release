from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import brentq
from src.preprocessing import load_data,initial_population,BOUNDS,POISSON
from src.physics_model import phase_velocity
from src.physics_model import var,dnka,dltar4,gtsolh
from src.solvers import objective,competitive_swarm,shrink,physical_model
ROOT=Path(__file__).resolve().parents[2]

def test_public_acquisition_constants_shapes_and_exact_author_conversion():
    readme=(ROOT/'README.md').read_text(encoding='utf8')
    assert '| Poisson ratio | 0.30,0.30,0.49,0.49 |' in readme
    assert '| Density | 1.95 in every layer |' in readme
    assert_allclose(POISSON,[.3,.3,.49,.49],rtol=0,atol=0)
    data=load_data()
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as z:author=z['author_observations']
    assert_allclose(data['period'],author[:,0]/author[:,1],rtol=0,atol=0)
    assert_allclose(data['phase_velocity'],author[:,1]/1000,rtol=0,atol=0)
    assert_allclose(data['uncertainty'],(author[:,3]-author[:,2])/2000,rtol=0,atol=0)
    with np.load(ROOT/'data/raw_data.npz') as z:
        for key in z.files:
            assert z[key].shape==(1,30) and z[key].dtype==np.dtype('float64')
            assert '| '+key+' | (1,30) | float64 |' in readme
    parameters=np.r_[.001,.002,.008,.1,.14,.18,.2,POISSON]
    model=physical_model(parameters)
    assert_allclose(model[:,3],1.95,rtol=0,atol=0)
    assert_allclose(model[:,1],model[:,2]*np.sqrt((1-POISSON)/(.5-POISSON)),rtol=0,atol=0)

def test_original_compound_entries_eigenfunctions_and_surface_determinant():
    with np.load(ROOT/'test/fixtures/upstream_operators.npz') as z:
        model=z['model'];omega=2*np.pi/.04
        for i,c in enumerate([.09,.15,.30]):
            for actual,expected in [(np.array(var(*z[f'var_args_{i}'])),z[f'var_{i}']),
                                    (dnka(*z[f'matrix_args_{i}'],np.empty((5,5))),z[f'matrix_{i}'])]:
                assert_allclose(actual,expected,rtol=0,atol=1e-12*max(1.,np.max(abs(expected))))
            assert_allclose(dltar4(omega/c,omega,*model.T,-1,np.empty((5,5))),z[f'secular_{i}'],rtol=0,atol=1e-12)
        assert_allclose(gtsolh(model[0,1],model[0,2]),z['homogeneous_initial'],rtol=0,atol=1e-12)

def test_frozen_original_forward_objective_and_two_complete_updates():
    data=load_data()
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as z:
        assert_allclose(data['period'],z['period'],rtol=0,atol=0)
        assert_allclose(phase_velocity(data['period'],*z['model'].T),z['prediction'],rtol=1e-6,atol=0)
        population=initial_population(np.random.RandomState(24713))
        assert_allclose(population,z['initial_population'],rtol=0,atol=1e-12)
        values=np.array([objective(p,data) for p in z['trial_parameters']])
        assert_allclose(values,z['trial_objectives'],rtol=0,atol=1e-5)
        _,stages=competitive_swarm(data,maxiter=2)
        assert_allclose(stages['parameters'][1],z['second_population'],rtol=0,atol=1e-12)
        assert_allclose(stages['global_objectives'],z['global_misfits'][:2],rtol=0,atol=1e-5)

def test_independent_homogeneous_rayleigh_polynomial_and_depth_scaling():
    nu=.3;vs=.2;vp=vs*np.sqrt((1-nu)/(.5-nu));ratio=(vs/vp)**2
    secular=lambda q:(2-q*q)**2-4*np.sqrt(1-q*q)*np.sqrt(1-ratio*q*q)
    expected=vs*brentq(secular,.5,.999)
    periods=np.array([.02,.04,.08]);h=np.array([.001,.003,.006,1.])
    actual=phase_velocity(periods,h,np.full(4,vp),np.full(4,vs),np.full(4,1.95))
    assert_allclose(actual,expected,rtol=1e-6)
    data=load_data()
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as z:model=z['model']
    original=phase_velocity(data['period'],*model.T)
    scaled=model.copy();scaled[:,0]*=3
    assert_allclose(phase_velocity(data['period']*3,*scaled.T),original,rtol=1e-6)

def test_shrinking_repair_and_paper_radius_coordinate_difference():
    position=np.array([.5,.5]);velocity=np.array([1.,-.25])
    x,v=shrink(position,velocity,np.zeros(2),np.ones(2))
    assert_allclose(x,[1.,.375]);assert_allclose(v,[.5,-.125])
    displacement=np.array([.1,.2]);span=np.array([1.,2.])
    source=lambda d:np.linalg.norm(d)/np.sqrt(8.)
    paper=lambda d,s:np.linalg.norm(d)/np.linalg.norm(s)
    assert_allclose(source(100*displacement),100*source(displacement))
    assert_allclose(paper(100*displacement,100*span),paper(displacement,span))
    assert not np.isclose(source(displacement),paper(displacement,span))

def test_fresh_complete_cli_output_and_all_seven_unknowns():
    with np.load(ROOT/'output/reconstruction.npz') as z, np.load(ROOT/'evaluation/baseline_reference.npz') as ref:
        for key in ['reconstruction','thickness']:assert_allclose(z[key],ref[key],rtol=1e-6,atol=1e-6)
    assert BOUNDS.shape==(11,2)
    assert np.count_nonzero(np.diff(BOUNDS,axis=1))==7
