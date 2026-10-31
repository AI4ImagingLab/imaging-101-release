"""Layer-averaged radioactive decay, with mass measured downward from the surface."""
import numpy as np
DECAY_CONSTANT=0.03118  # yr^-1, author nuclear-data convention

def layer_activity(initial_activity,rate,mass_thickness,previous_age):
    """Return mean Bq/kg across a layer deposited at constant g/cm^2/yr rate."""
    duration=mass_thickness/rate
    x=DECAY_CONSTANT*duration
    return initial_activity*np.exp(-DECAY_CONSTANT*previous_age)*(1-np.exp(-x))/x

def physical_profile(mass_depth,ages,initial_activity):
    """Forward observation for a supplied chronology and initial concentration."""
    duration=np.diff(np.concatenate(([0.],ages)))
    dm=np.diff(np.concatenate(([0.],mass_depth)))
    rate=dm/duration
    previous=np.concatenate(([0.],ages[:-1]))
    return layer_activity(initial_activity,rate,dm,previous)
