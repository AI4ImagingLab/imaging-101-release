"""Generate test fixtures for mri_dynamic_dce task."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.physics_model import (
    fft2c, ifft2c, forward_single, adjoint_single,
    forward_dynamic, adjoint_dynamic, normal_operator_dynamic,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "evaluation", "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Use seeded RNG for reproducibility
rng = np.random.RandomState(42)

N = 16  # Image size (small for test fixtures)
T = 3   # Number of time frames

# ---- Single image (N, N) real ----
input_image = rng.randn(N, N).astype(np.float64)

# ---- Dynamic images (T, N, N) real ----
input_images = rng.randn(T, N, N).astype(np.float64)

# ---- Single mask (N, N) binary ----
input_mask_single = np.zeros((N, N), dtype=np.float64)
# Sample every other column (Cartesian-like)
input_mask_single[:, ::2] = 1.0

# ---- Dynamic masks (T, N, N) binary ----
input_masks = np.zeros((T, N, N), dtype=np.float64)
for t in range(T):
    # Different sampling pattern per frame
    input_masks[t, :, t::3] = 1.0

# ---- Compute outputs ----
output_fft2c = fft2c(input_image)
output_ifft2c = ifft2c(output_fft2c)

output_forward_single = forward_single(input_image, input_mask_single)
output_adjoint_single = adjoint_single(output_forward_single)

output_forward_dynamic = forward_dynamic(input_images, input_masks)
output_adjoint_dynamic = adjoint_dynamic(output_forward_dynamic)

output_normal_dynamic = normal_operator_dynamic(input_images, input_masks)

# ---- Save all in one file ----
np.savez(
    os.path.join(FIXTURE_DIR, "physics_model.npz"),
    input_image=input_image,
    input_images=input_images,
    input_mask_single=input_mask_single,
    input_masks=input_masks,
    output_fft2c=output_fft2c,
    output_ifft2c=output_ifft2c,
    output_forward_single=output_forward_single,
    output_adjoint_single=output_adjoint_single,
    output_forward_dynamic=output_forward_dynamic,
    output_adjoint_dynamic=output_adjoint_dynamic,
    output_normal_dynamic=output_normal_dynamic,
)

print("Saved physics_model.npz with keys:")
for name in ['input_image', 'input_images', 'input_mask_single', 'input_masks',
             'output_fft2c', 'output_ifft2c', 'output_forward_single',
             'output_adjoint_single', 'output_forward_dynamic',
             'output_adjoint_dynamic', 'output_normal_dynamic']:
    arr = eval(name)
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

# ---- Verify roundtrip ----
rt = ifft2c(fft2c(input_image))
np.testing.assert_allclose(rt.real, input_image, atol=1e-10)
print("\nVerification passed!")
print("All mri_dynamic_dce fixtures generated successfully!")
