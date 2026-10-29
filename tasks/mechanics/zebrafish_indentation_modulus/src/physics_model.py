"""Explicit corrected spherical-contact mechanics, adapted from nanite GPL-3.0."""
import numpy as np
RADIUS = 18.64e-6
POISSON = .5
WEIGHT_DISTANCE = 2e-6

def contact_force(position, modulus, contact, radius=RADIUS, poisson=POISSON, corrected=True):
    """Return force [N]; positions/contact/radius [m], modulus [Pa]."""
    depth = contact - np.asarray(position)
    positive = depth > 0
    bb = np.zeros_like(depth)
    d = depth[positive]
    q = d / radius
    series = 1 - q/10 - q**2/840 + 11*q**3/15120 + 1357*q**4/6652800 if corrected else 1
    bb[positive] = d**1.5 * series
    return (4/3 * modulus/(1-poisson**2)*np.sqrt(radius)) * bb

def weights(position, contact, width=WEIGHT_DISTANCE):
    """Linear suppression around contact; unit weights when width=0."""
    return np.minimum(np.abs(position-contact)/width,1) if width else np.ones_like(position)

def residual(parameters, position, force, width=WEIGHT_DISTANCE, corrected=True):
    """Weighted force-minus-model residual [N], parameters=(E,c)."""
    modulus, contact = parameters
    return (force-contact_force(position,modulus,contact,corrected=corrected))*weights(position,contact,width)
