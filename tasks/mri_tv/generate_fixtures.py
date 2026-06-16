"""Generate test fixtures for mri_tv task."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.physics_model import fft2c, ifft2c, forward_operator, adjoint_operator, generate_undersampling_mask
from src.solvers import tv_reconstruct_single

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIXTURE_PM = os.path.join(os.path.dirname(__file__), "evaluation", "fixtures", "physics_model")
FIXTURE_SOLV = os.path.join(os.path.dirname(__file__), "evaluation", "fixtures", "solvers")

os.makedirs(FIXTURE_PM, exist_ok=True)
os.makedirs(FIXTURE_SOLV, exist_ok=True)

# Load raw data
raw = np.load(os.path.join(DATA_DIR, "raw_data.npz"))
masked_kspace_all = raw["masked_kspace"]       # (1, 15, 320, 320)
sensitivity_maps_all = raw["sensitivity_maps"]  # (1, 15, 320, 320)

masked_kspace_0 = masked_kspace_all[0]   # (15, 320, 320)
smaps_0 = sensitivity_maps_all[0]        # (15, 320, 320)

# ---- input_fft2c.npz / output_fft2c.npz ----
# Use a 320x320 complex random image (seeded for reproducibility)
rng = np.random.RandomState(42)
image_fft = (rng.randn(320, 320) + 1j * rng.randn(320, 320)).astype(np.complex128)

kspace_fft = fft2c(image_fft)

np.savez(os.path.join(FIXTURE_PM, "input_fft2c.npz"), image=image_fft)
np.savez(os.path.join(FIXTURE_PM, "output_fft2c.npz"), kspace=kspace_fft)
print("Saved input_fft2c.npz and output_fft2c.npz")

# ---- input_forward.npz / output_forward.npz ----
# Generate a mask using mri_tv's default params
mask = generate_undersampling_mask(320, 8, 0.04, "random", seed=0)

# Use a random complex image as forward input (ground truth may not exist as 320x320)
gt = np.load(os.path.join(DATA_DIR, "ground_truth.npz"))
# Check what keys are available
print("Ground truth keys:", gt.files)
for k in gt.files:
    print(f"  {k}: {gt[k].shape} {gt[k].dtype}")

# Use the adjoint of the first sample's k-space as a reasonable image
image_forward = adjoint_operator(masked_kspace_0, smaps_0)  # (320, 320) complex

masked_ksp = forward_operator(image_forward, smaps_0, mask)

np.savez(os.path.join(FIXTURE_PM, "input_forward.npz"),
         image=image_forward, sensitivity_maps=smaps_0, mask=mask)
np.savez(os.path.join(FIXTURE_PM, "output_forward.npz"),
         masked_kspace=masked_ksp)
print("Saved input_forward.npz and output_forward.npz")

# ---- output_adjoint.npz ----
adjoint_img = adjoint_operator(masked_kspace_0, smaps_0)
np.savez(os.path.join(FIXTURE_PM, "output_adjoint.npz"), image=adjoint_img)
print("Saved output_adjoint.npz")

# ---- output_mask.npz ----
mask_fixture = generate_undersampling_mask(320, 8, 0.04, "random", seed=0)
np.savez(os.path.join(FIXTURE_PM, "output_mask.npz"), mask=mask_fixture)
print("Saved output_mask.npz")

# ---- solvers: output_tv_recon_sample0.npz ----
print("Running TV reconstruction (this may take a while for 320x320)...")
recon = tv_reconstruct_single(masked_kspace_0, smaps_0, lamda=1e-4)
np.savez(os.path.join(FIXTURE_SOLV, "output_tv_recon_sample0.npz"), reconstruction=recon)
print("Saved output_tv_recon_sample0.npz")

print("\nAll mri_tv fixtures generated successfully!")
