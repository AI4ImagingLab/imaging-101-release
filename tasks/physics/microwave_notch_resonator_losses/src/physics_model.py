"""Independent explicit transmission-line notch forward model."""
import numpy as np

def forward(frequency, resonance, loaded_quality, coupling_magnitude, coupling_phase, amplitude=1., phase=0., delay=0.):
    detuning=(frequency-resonance)/resonance
    return amplitude*np.exp(1j*phase-2j*np.pi*frequency*delay)*(1-loaded_quality*np.exp(1j*coupling_phase)/coupling_magnitude/(1+2j*loaded_quality*detuning))

def dissipation_quality(loaded_quality, coupling_magnitude, coupling_phase):
    coupling=coupling_magnitude/np.cos(coupling_phase)
    return coupling,1/(1/loaded_quality-1/coupling)
