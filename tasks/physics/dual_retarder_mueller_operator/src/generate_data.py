"""Reproduce the author-native synthetic example; never called by solver.

The unchanged original notebook constructs one random physical retarder times
an ideal linear polarizer, then24angles linspace(0,pi,24), two known ideal
quarter-wave rotating modulators with speed ratio5, unit unpolarized incident
Stokes intensity, unit detector gain, zero offset/no added detector noise.
Private capture seed20261026 fixes the original notebook's three legacy NumPy
random draws for repeatability. This module reproduces that authored experiment
with local optics solely for validation; published raw observations/truth were
captured independently from upstream before extraction. No seed, target or
target-derived calibration is in solver-visible data. Alternate generated data
must not overwrite the bound acquisition/reference during normal execution.
"""
import numpy as np
from .physics_model import linear_polarizer,linear_retarder,forward

def author_example(seed=20261026):
    rng=np.random.RandomState(seed)
    angle,retardance,polarizer_angle=rng.random_sample(3)
    truth=linear_retarder(angle,retardance)@linear_polarizer(polarizer_angle)
    angles=np.linspace(0,np.pi,24)
    return {'angles':angles[None],'power':forward(truth,angles)[None]},truth
