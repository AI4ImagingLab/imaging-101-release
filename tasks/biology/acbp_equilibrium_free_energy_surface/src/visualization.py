"""Physical diagnostics; no plotting backend side effects."""
import numpy as np
from .physics_model import free_energy
def denaturant_midpoint(temperature,parameters):
 p=np.asarray(parameters);return free_energy(temperature,0.,p)/p[3]
def relative_l2(prediction,observation):return float(np.linalg.norm(prediction-observation)/np.linalg.norm(observation))
