from pathlib import Path
import numpy as np
from src.physics_model import load_ions

def test_frozen_original_atomic_stages():
    root=Path(__file__).parents[2]
    ions=load_ions(root/'data/raw_data.npz')
    with np.load(root/'test/fixtures/upstream_atomic.npz',allow_pickle=False) as a:
        for key,ion in ions.items():
            t=np.array([7500.,10000.,15000.]);n=np.array([100.,1000.,6000.])
            for suffix,value in [('omega',ion.omega(t)),('rates',ion.rates(t)),('populations',ion.populations(t,n)),('emissivities',ion.emissivities(t,n))]:
                expected=a[key+'_'+suffix]
                # Original getOmega returns every tabulated collision level,
                # while the model uses min(A-levels, collision-levels).
                if suffix=='omega':expected=expected[:ion.n,:ion.n]
                np.testing.assert_allclose(value,expected,rtol=1e-10,atol=0,err_msg=key+suffix)
            pop=ion.populations(t,n)
            np.testing.assert_allclose(pop.sum(axis=0),1.,rtol=1e-13)
            assert np.all(pop>=0)
