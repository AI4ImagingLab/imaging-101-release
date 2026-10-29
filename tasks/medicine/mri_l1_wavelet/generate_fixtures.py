"""Generate test fixtures for mri_l1_wavelet task."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.physics_model import fft2c, ifft2c, forward_operator, adjoint_operator, generate_undersampling_mask
from src.solvers import l1_wavelet_reconstruct_single

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIXTURE_PM = os.path.join(os.path.dirname(__file__), "evaluation", "fixtures", "physics_model")
FIXTURE_SOLV = os.path.join(os.path.dirname(__file__), "evaluation", "fixtures", "solvers")

os.makedirs(FIXTURE_PM, exist_ok=True)
os.makedirs(FIXTURE_SOLV, exist_ok=True)

# Load raw data
raw = np.load(os.path.join(DATA_DIR, "raw_data.npz"))
masked_kspace_all = raw["masked_kspace"]       # (1, 8, 128, 128)
sensitivity_maps_all = raw["sensitivity_maps"]  # (1, 8, 128, 128)

masked_kspace_0 = masked_kspace_all[0]   # (8, 128, 128)
smaps_0 = sensitivity_maps_all[0]        # (8, 128, 128)

# ---- input_fft2c.npz / output_fft2c.npz ----
# Use a 128x128 complex random image (seeded for reproducibility)
rng = np.random.RandomState(42)
image_fft = (rng.randn(128, 128) + 1j * rng.randn(128, 128)).astype(np.complex128)

kspace_fft = fft2c(image_fft)

np.savez(os.path.join(FIXTURE_PM, "input_fft2c.npz"), image=image_fft)
np.savez(os.path.join(FIXTURE_PM, "output_fft2c.npz"), kspace=kspace_fft)
print("Saved input_fft2c.npz and output_fft2c.npz")

# ---- input_forward.npz / output_forward.npz ----
# Generate a mask
mask = generate_undersampling_mask(128, 8, 0.08, "random", seed=0)

# Load ground truth phantom as image
gt = np.load(os.path.join(DATA_DIR, "ground_truth.npz"))
image_forward = gt["phantom"][0, 0]  # (128, 128) complex

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
mask_fixture = generate_undersampling_mask(128, 8, 0.08, "random", seed=0)
np.savez(os.path.join(FIXTURE_PM, "output_mask.npz"), mask=mask_fixture)
print("Saved output_mask.npz")

# ---- solvers: output_l1wav_recon_sample0.npz ----
print("Running L1 wavelet reconstruction (this may take a moment)...")
recon = l1_wavelet_reconstruct_single(masked_kspace_0, smaps_0, lamda=1e-3, max_iter=100)
np.savez(os.path.join(FIXTURE_SOLV, "output_l1wav_recon_sample0.npz"), reconstruction=recon)
print("Saved output_l1wav_recon_sample0.npz")

print("\nAll mri_l1_wavelet fixtures generated successfully!")
