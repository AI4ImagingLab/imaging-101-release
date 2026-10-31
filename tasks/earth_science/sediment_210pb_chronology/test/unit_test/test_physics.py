import numpy as np
from scipy.integrate import quad
from src.physics_model import layer_activity,physical_profile
from src.preprocessing import canonical_proposals

def test_independent_decay_integral_and_units():
    a0=640.;w=.04;dm=.16;previous=8.
    numerical=quad(lambda time:a0*np.exp(-.03118*time),previous,previous+dm/w)[0]/(dm/w)
    np.testing.assert_allclose(layer_activity(a0,w,dm,previous),numerical,rtol=1e-13)

def test_deposition_time_and_initial_activity_forward():
    mass=np.array([.2,.5]);age=np.array([5.,11.]);a0=np.array([100.,200.])
    result=physical_profile(mass,age,a0)
    assert np.all(result<a0)
    assert np.all(result>0)

def test_stochastic_canonical_support_and_distribution():
    first,second=canonical_proposals(seed=17)
    np.testing.assert_allclose(np.sort(first),np.sort(second))
    assert abs(first.mean())<1e-12
    assert .8<first.std()<1.1
    assert np.max(abs(first))<2
    assert not np.array_equal(first,second)

def test_readme_acquisition_constants_and_output_keys():
    from pathlib import Path
    from src.physics_model import DECAY_CONSTANT
    text=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    assert f'| Decay constant λ | {DECAY_CONSTANT} | yr⁻¹ |' in text
    assert '| Number of layers | 14 |' in text
    assert '| Collection year | 2007 |' in text
    for key in ['reconstruction','sedimentation_rate','initial_activity','modeled_activity']:
        assert f'| `{key}` | `(1, 14)` |' in text
