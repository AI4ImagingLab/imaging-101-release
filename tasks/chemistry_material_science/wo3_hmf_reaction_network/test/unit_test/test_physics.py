import numpy as np
from src.physics_model import concentrations,charge,derivatives,FARADAY,VOLUME_L,ELECTRON_WEIGHTS
from src.solvers import relative_residual

def test_independent_mass_balance():
    rng=np.random.default_rng(384)
    for _ in range(20):
        c=rng.uniform(0,5000,15);rates=rng.uniform(0,.3,11)
        assert abs(derivatives(c,0,rates).sum())<1e-10

def test_analytic_two_electron_path():
    initial=np.zeros(15);initial[0]=100
    rates=np.zeros(11);rates[0]=.2
    t=np.linspace(0,10,100)
    c=concentrations(t,initial,rates)
    np.testing.assert_allclose(c[:,0],100*np.exp(-.2*t),rtol=1e-8)
    np.testing.assert_allclose(c.sum(axis=1),100,rtol=2e-9)
    np.testing.assert_allclose(charge(c),2*FARADAY*VOLUME_L*1e-6*100*(1-np.exp(-.2*t)),rtol=1e-8,atol=1e-12)
    assert c.min()>=-1e-8

def test_relative_known_case():
    np.testing.assert_allclose(relative_residual(np.array([0.,2.,3.]),np.array([0.,1.,9.])),[0,1/3,-.5],rtol=1e-15)

def test_all_known_half_reactions_and_path_independence():
    # HMF->DFF/HMFCA, DFF/HMFCA->FFCA, FFCA->FDCA each removes2e.
    for source,target in [(0,1),(0,2),(1,3),(2,3),(3,4)]:
        assert ELECTRON_WEIGHTS[target]-ELECTRON_WEIGHTS[source]==2
    assert sum(ELECTRON_WEIGHTS[b]-ELECTRON_WEIGHTS[a] for a,b in [(0,1),(1,3),(3,4)])==6
    assert sum(ELECTRON_WEIGHTS[b]-ELECTRON_WEIGHTS[a] for a,b in [(0,2),(2,3),(3,4)])==6
    # Balance C,H,O and electric charge for all known molecular half-reactions.
    formula=np.array([[6,6,3],[6,4,3],[6,6,4],[6,4,4],[6,4,5]])
    for a,b,waters in [(0,1,0),(0,2,1),(1,3,1),(2,3,0),(3,4,1)]:
        np.testing.assert_array_equal(formula[a]+waters*np.array([0,2,1]),formula[b]+np.array([0,2,0]))

def test_acquired_charge_zero_with_initial_oxidized_species():
    initial=np.arange(1.,16.)
    curves=concentrations(np.linspace(0,10,50),initial,np.full(11,.01))
    assert charge(curves)[0]==0
    assert np.all(np.diff(charge(curves))>=-1e-10)
