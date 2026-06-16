"""Generate test fixtures for mri_t2_mapping task."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.physics_model import mono_exponential_signal

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "evaluation", "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)

# Create test parameters
# Use a small array of M0 and T2 values representing different tissue types
# The test_physics_model.py uses fix['param_M0'][0] and fix['param_T2'][0] as scalars
# So param_M0 and param_T2 should be 1D arrays where [0] gives a scalar

param_M0 = np.array([1.0, 0.8, 1.2], dtype=np.float64)
param_T2 = np.array([80.0, 50.0, 120.0], dtype=np.float64)

# Echo times - typical multi-echo spin echo
input_TE = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float64)

# The test does: mono_exponential_signal(np.array(M0), np.array(T2), TE)
# where M0 = fix['param_M0'][0] = 1.0 (scalar) and T2 = fix['param_T2'][0] = 80.0 (scalar)
# result shape: (1, 8) -> ravel() -> (8,)
# expected = fix['output_signal_clean'] should match that flattened output

M0_scalar = param_M0[0]  # 1.0
T2_scalar = param_T2[0]  # 80.0
output_signal_clean = mono_exponential_signal(np.array(M0_scalar), np.array(T2_scalar), input_TE)
output_signal_clean = output_signal_clean.ravel()

np.savez(
    os.path.join(FIXTURE_DIR, "physics_model_fixtures.npz"),
    param_M0=param_M0,
    param_T2=param_T2,
    input_TE=input_TE,
    output_signal_clean=output_signal_clean,
)

print(f"param_M0: {param_M0}")
print(f"param_T2: {param_T2}")
print(f"input_TE: {input_TE}")
print(f"output_signal_clean: {output_signal_clean}")
print(f"output_signal_clean shape: {output_signal_clean.shape}")

# Verify
result = mono_exponential_signal(np.array(M0_scalar), np.array(T2_scalar), input_TE)
np.testing.assert_allclose(result.ravel(), output_signal_clean, rtol=1e-10)
print("\nVerification passed!")
print("All mri_t2_mapping fixtures generated successfully!")
