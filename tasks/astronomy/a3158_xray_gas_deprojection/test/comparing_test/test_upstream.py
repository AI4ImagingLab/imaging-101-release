from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_data, background_subtraction
from src.solvers import reconstruct, onion_peeling
from src.physics_model import projection_volumes, count_to_hydrogen_emission_measure, ELECTRON_TO_HYDROGEN, LUMINOSITY_MPC, MPC_CM, REDSHIFT, COUNT_FACTOR

FIX=Path(__file__).parents[1]/'fixtures/upstream_stages.npz'

def test_native_image_profile_and_full_original_pipeline():
    data=load_data();out,stages=reconstruct(data)
    with np.load(FIX) as z:
        m=stages['measured']
        assert_allclose(m['center'],[271,280],atol=0)
        for local,source in [('profile_raw','profile_raw'),('error_raw','profile_raw_error'),('area','profile_area'),('exposure','profile_effexp')]:
            assert_allclose(m[local],z[source],rtol=1e-12,atol=1e-16)
        assert_allclose(m['background'],z['background_value'],rtol=.001)
        assert_allclose(m['background_error'],z['background_error'],rtol=.001)
        assert_allclose(stages['volume'],z['solve_matrix'],rtol=1e-12)
        # Tight original optimizer and exact weighted solution agree well inside
        # the predeclared density .002 / emissivity .004 parity tolerances.
        assert_allclose(out['reconstruction'][0],z['output_density'],rtol=.002)
        assert_allclose(stages['historical_smoothed_emissivity'],z['output_emissivity'],rtol=.004)
        assert_allclose(out['emissivity'],out['reconstruction']**2,rtol=0,atol=0)

def test_original_solver_stages_independent_of_background_fit():
    data=load_data()
    with np.load(FIX) as z:
        density,uncertainty,q,s=onion_peeling(z['profile_backsub'],z['profile_backsub_error'],data['radial_edges'])
        assert_allclose(s['rhs'],z['solve_rhs'],rtol=1e-11)
        assert_allclose(s['samples'],z['solve_output'],rtol=1e-11,atol=1e-18)
        assert_allclose(density,z['output_density'],rtol=1e-11)
        assert_allclose(q,z['output_emissivity'],rtol=1e-11)

def test_uniform_sphere_and_discrete_shell_inverse():
    # Independent analytic cylinder intersection of a unit-radius sphere.
    edges=np.linspace(0,1,11);v=projection_volumes(edges[:-1],edges[1:])
    projected=v@np.ones(10)
    analytic=4*np.pi/3*((1-edges[:-1]**2)**1.5-(1-edges[1:]**2)**1.5)
    assert_allclose(projected,analytic,rtol=1e-13,atol=1e-15)
    assert_allclose(v.sum(axis=0),4*np.pi/3*np.diff(edges**3),rtol=1e-12)
    q=np.linspace(.2,1.3,10)
    assert_allclose(np.linalg.solve(v,v@q),q,rtol=1e-13)
    assert_allclose(np.dot(v@q,q),np.dot(q,v.T@q),rtol=1e-14)

def test_xspec_hydrogen_not_electron_conversion():
    da=LUMINOSITY_MPC*MPC_CM/(1+REDSHIFT)**2
    nh=.003;volume=2e70
    ne=ELECTRON_TO_HYDROGEN*nh
    norm=1e-14*ne*nh*volume/(4*np.pi*da**2*(1+REDSHIFT)**2)
    counts=COUNT_FACTOR*norm
    assert_allclose(counts*count_to_hydrogen_emission_measure(),nh**2*volume,rtol=2e-15)
    assert not np.isclose(nh,ne,rtol=.01)
