"""Explicit Hansen-Law forward model and shared state propagation.

Adapted from PyAbel v0.9.1 abel/hansenlaw.py, MIT License.
Copyright (c) 2015 The PyAbel Team. See ../test/validation.md for the full MIT terms.
Original implementation: Stephen Gibson; vectorization: Dan Hickstein and
Roman Yurchak; first-order hold rewrite with Eric Hansen (2018).
"""
import numpy as np

H = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
LAMBDA = np.array([0., -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9, -47391.1])


def validate_half_image(image, dr, hold_order):
    """Validate a real finite vector or (z, radius) float-convertible array."""
    array = np.asarray(image)
    if (array.ndim not in (1, 2) or array.shape[-1] < 3 or array.size == 0
            or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array)):
        raise ValueError('Expected a real vector or 2-D half-image with >=3 columns')
    if not np.isfinite(array).all():
        raise ValueError('Input must be finite')
    if not np.isfinite(dr) or dr <= 0 or hold_order not in (0, 1):
        raise ValueError('dr must be positive and finite; hold_order must be 0 or 1')
    return np.atleast_2d(array.astype(np.float64)), array.ndim == 1


def state_integrals(n, a):
    """Integrate (epsilon/r)^(lambda+a) on a descending radial interval."""
    n = np.asarray(n)
    ratio = n/(n-1)
    result = np.empty((n.size, H.size))
    start = 0
    if a == 0:
        result[:, 0] = -np.log(ratio)
        start = 1
    for k in range(start, H.size):
        exponent = LAMBDA[k]+a
        result[:, k] = (n-1)**a*(1-ratio**exponent)/exponent
    return result


def state_coefficients(cols, direction='inverse', hold_order=1):
    """Return (Phi, B0, B1), each shape (cols-2, 9), in float64."""
    if cols < 3 or direction not in ('forward', 'inverse') or hold_order not in (0, 1):
        raise ValueError('Invalid state-equation configuration')
    n = np.arange(cols-1, 1, -1)
    phi = np.empty((n.size, H.size))
    for k, exponent in enumerate(LAMBDA):
        phi[:, k] = (n/(n-1))**exponent
    a = 1 if direction == 'forward' else 0
    gamma0 = state_integrals(n, a)*H
    if hold_order == 0:
        return phi, gamma0*0, gamma0
    gamma1 = state_integrals(n, a+1)*H
    return phi, gamma1-gamma0*(n-1)[:, None], gamma0*n[:, None]-gamma1


def propagate(drive, coefficients):
    """Zero-initialize nine states and sweep from outer radius to axis."""
    rows, cols = drive.shape
    phi, b0, b1 = coefficients
    output = np.empty_like(drive)
    state = np.zeros((H.size, rows))
    for index, col in enumerate(range(cols-2, 0, -1)):
        state = (phi[index, :, None]*state + b0[index, :, None]*drive[:, col+1]
                 + b1[index, :, None]*drive[:, col])
        output[:, col] = state.sum(axis=0)
    # Preserve the selected upstream endpoint convention, including edge copying.
    output[:, 0] = output[:, 1]
    output[:, -1] = output[:, -2]
    return output


def forward_transform(emission, dr=0.025, hold_order=1):
    """Project a (z,r) emission density into (z,y) radiance; multiply by mm."""
    image, vector = validate_half_image(emission, dr, hold_order)
    drive = -2*dr*np.pi*image
    output = propagate(drive, state_coefficients(image.shape[1], 'forward', hold_order))
    return output[0] if vector else output
