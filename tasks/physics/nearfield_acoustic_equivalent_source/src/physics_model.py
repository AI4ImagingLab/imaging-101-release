"""Explicit free-field equivalent-monopole model, exp(+i omega t) convention.

Extracted from Pagavino/Zotter2025 original source, MIT Manuel Pagavino2025.
G units are Pa*s/m^3. Pressure in author units U gives relative q units
U*m^3/(Pa*s); the dataset does not establish absolute Pa or volume velocity.
"""
import numpy as np
FREQUENCY_HZ=1150.;SOUND_SPEED=343.;AIR_DENSITY=1.2
WIDTH_M=.825;HEIGHT_M=.6;DISTANCE_M=.025
SOURCE_SHAPE=(40,50);MEASUREMENT_SHAPE=(9,12)
def coordinates(shape,width_m=WIDTH_M,height_m=HEIGHT_M,z_m=0.):
    """Equispaced inclusive grid, row-major flattening, coordinates in m."""
    x=np.linspace(0.,width_m,shape[1]);y=np.linspace(0.,height_m,shape[0])
    xx,yy=np.meshgrid(x,y)
    return np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,z_m)))
def propagator(distance_m=DISTANCE_M,frequency_hz=FREQUENCY_HZ):
    """Return complex64 G(108,2000), including i*omega*rho and free-space spreading."""
    source=coordinates(SOURCE_SHAPE);receiver=coordinates(MEASUREMENT_SHAPE,z_m=distance_m)
    radius=np.sqrt(np.sum((receiver[:,None,:]-source[None,:,:])**2,axis=-1))
    if (radius<=0).any():raise ValueError('receiver must not coincide with equivalent source')
    omega=2*np.pi*frequency_hz;k=omega/SOUND_SPEED
    return (np.exp(-1j*k*radius)/(4*np.pi*radius)*(1j*omega*AIR_DENSITY)).astype(np.complex64)
def forward(q,distance_m=DISTANCE_M,frequency_hz=FREQUENCY_HZ):
    """Project complex relative source strengths (40,50) to observed pressure (9,12)."""
    return (propagator(distance_m,frequency_hz)@np.asarray(q).reshape(-1)).reshape(MEASUREMENT_SHAPE)
def adjoint(pressure,distance_m=DISTANCE_M):
    """Hermitian transpose, not a physical time-reversed propagation shortcut."""
    return (propagator(distance_m).conj().T@np.asarray(pressure).ravel()).reshape(SOURCE_SHAPE)
