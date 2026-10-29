"""Explicit N2H+ LTE radiative transfer, adapted from MIT pyspeckit."""
import numpy as np
C_KMS = 299792.458
H_OVER_K = 6.62607015e-34 / 1.380649e-23
REST_FREQUENCY = 93176265000.0
CENTRAL_FREQUENCY = 93173763700.0
BACKGROUND_K = 2.73
OFFSETS = np.array([-7.9930]*3 + [-.6112]*3 + [0., .9533, .9533] + [5.5371]*3 + [5.9704]*2 + [6.9238])
STRENGTHS = np.array([.025957,.065372,.019779,.004376,.034890,.071844,.259259,.156480,.028705,.041361,.013309,.056442,.156482,.028705,.037038])
FREQUENCIES = CENTRAL_FREQUENCY * (1-OFFSETS/C_KMS)
NORMALIZATION = sum(STRENGTHS)

def optical_depth(frequency, parameters):
    """Dimensionless optical depth at frequency (Hz), parameters=(K,1,km/s,km/s)."""
    _, tau, velocity, sigma = np.asarray(parameters)
    frequency = np.asarray(frequency, dtype=np.float64)
    if sigma <= 0:
        raise ValueError('Dispersion must be positive')
    total = np.zeros_like(frequency)
    for offset, center, strength in zip(OFFSETS, FREQUENCIES, STRENGTHS):
        # Preserve upstream component-specific width and Doppler-shift convention.
        shifted_line = (1-offset/C_KMS)*center
        width = abs(sigma/C_KMS*shifted_line)
        shift = velocity/C_KMS*shifted_line
        total += tau*strength/NORMALIZATION*np.exp(-(frequency+shift-center)**2/(2*width**2))
    return total

def brightness(frequency, parameters):
    """Background-subtracted brightness K, shape matching frequency."""
    temperature = np.asarray(parameters)[0]
    if temperature <= 0:
        raise ValueError('Excitation temperature must be positive')
    t0 = H_OVER_K*np.asarray(frequency)
    source = t0*(1/(np.exp(t0/temperature)-1)-1/(np.exp(t0/BACKGROUND_K)-1))
    return (1-np.exp(-optical_depth(frequency, parameters)))*source

def predict_cube(frequency, parameters):
    """Map (4,Y,X) physical parameters into a (F,Y,X) brightness cube."""
    out = np.empty((len(frequency),)+parameters.shape[1:], dtype=np.float64)
    for y, x in np.ndindex(parameters.shape[1:]):
        out[:,y,x] = brightness(frequency, parameters[:,y,x])
    return out
