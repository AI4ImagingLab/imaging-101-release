"""Shared diagnostic numbers; plotting belongs in the active notebook."""
import numpy as np
def relative_error(prediction,reference):return float(np.max(np.abs(prediction-reference)/np.abs(reference)))
def residual_rms(residual):return float(np.sqrt(np.mean(np.asarray(residual)**2)))
