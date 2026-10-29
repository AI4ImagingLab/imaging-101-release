"""Maintainer-only author-defined simulated quantization.

The original real oak_leaf.png photograph (Stephen Heron, CC-BY-SA2.0)
is decoded exactly as the original author matplotlib.imread float32 RGB
image. Original test_dequantize.py then constructs lower=floor(6*f)/6,
upper=ceil(6*f)/6 at native512² resolution. The bound observations are
these intervals, not measured low-bit camera data. This helper implements
the same acquisition from a supplied ORIGINAL photo array, without RNG,
hidden object definitions, target-derived calibration or saved references.
The complete photo and capture source are private; the original image is
also the hidden actual truth. No solver path calls this generator.
"""
import numpy as np
from .physics_model import quantization_intervals

def generate_observations(original_image,bins=6):
    if original_image.dtype!=np.float32:raise ValueError('preserve original decoded float32 photo')
    lower,upper=quantization_intervals(original_image,bins)
    return {'lower':lower[None],'upper':upper[None]}
