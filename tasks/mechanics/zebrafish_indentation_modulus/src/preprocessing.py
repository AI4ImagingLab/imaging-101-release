"""Measured AFM calibration and baseline estimation (SI units).

Adapted from nanite, Paul Müller and Shada Abuhattum, GPL-3.0.
"""
import numpy as np

def contact_index(force):
    """Baseline-deviation estimate before the first force maximum."""
    clipped = force[:np.argmax(force)]
    base = clipped[:int(clipped.size * .1)]
    if base.size:
        mean = np.average(base)
        crossing = clipped - mean > 2 * np.max(np.abs(base - mean))
        index = int(np.argmax(crossing))
        if crossing[index]:
            return index
    return clipped.size // 2

def preprocess(force, height, spring_constant):
    """Return corrected tip [m], force [N], and absolute tip origin [m]."""
    tip = height + force / spring_constant
    first = contact_index(force)
    corrected = force - (np.average(force[:first]) if first else force[0])
    second = contact_index(corrected)
    origin = tip[second]
    return tip - origin, corrected, float(origin)

def load_data(path='data/raw_data.npz'):
    """Read batch-first arrays, without pickle or any hidden inputs."""
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k][0] for k in data.files}
