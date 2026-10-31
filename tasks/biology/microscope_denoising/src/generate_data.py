"""
Generate and prepare the microscope_denoising benchmark dataset.

Downloads the 2D fluorescence microscopy example data from Zenodo
(DOI: 10.5281/zenodo.7261163) and packages it into the standard npz format.

Dataset: TIRF fluorescence images of Microtubules acquired at low-SNR
conditions on a Multi-SIM system (Qiao et al., Nat. Commun. 2024).

Output layout:
  data/raw_data.npz      - key "measurements": shape (55, 502, 502), float32
                            55 noisy TIRF Microtubule images, background ~100 ADU,
                            average signal ~10 ADU above background.
  data/ground_truth.npz  - key "psf": shape (1, 32, 32), float32
                            Center-cropped, normalised PSF kernel used for
                            Stage 2 (deconvolution) training and evaluation.
  data/meta_data.json    - imaging parameters and noise model estimates.
"""

import os
import sys
import json
import zipfile
import urllib.request
import numpy as np
import tifffile

ZENODO_URL = (
    "https://zenodo.org/api/records/7261163/files/2D%20data.zip/content"
)
# generate_data.py lives in src/; TASK_DIR is one level up
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(_SRC_DIR)
DATA_DIR = os.path.join(TASK_DIR, 'data')
CACHE_ZIP = os.path.join(TASK_DIR, '2d_data.zip')

# Ensure task root is importable (so `from src.physics_model import...` works)
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)


def download_data(force=False):
    """Download 2D data zip from Zenodo if not already present."""
    if os.path.exists(CACHE_ZIP) and not force:
        print(f'Found cached zip: {CACHE_ZIP}')
        return
    print(f'Downloading 2D data from Zenodo (~12 MB)...')
    urllib.request.urlretrieve(ZENODO_URL, CACHE_ZIP,
                               reporthook=lambda b, bs, t:
                               print(f'\r  {b*bs/1e6:.1f}/{t/1e6:.1f} MB', end=''))
    print(f'\nSaved to {CACHE_ZIP}')


def extract_mt_images(zip_path):
    """
    Extract Microtubule train images from zip into memory.

    Returns
    -------
    images : np.ndarray, shape (55, 502, 502), float32
    """
    images = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = sorted([n for n in zf.namelist()
                        if 'Microtubule/train_data/' in n and n.endswith('.tif')])
        print(f'  Found {len(names)} Microtubule frames')
        for name in names:
            with zf.open(name) as f:
                img = tifffile.imread(f)
            images.append(img.astype(np.float32))
    return np.stack(images)   # (N, H, W)


def extract_psf(zip_path, kernel_size=32):
    """
    Extract and preprocess the Microtubule PSF from zip.

    Loads psf_emLambda525_dxy0.0313_NA1.3.tif, center-crops to kernel_size×kernel_size,
    and normalises to unit sum.

    Returns
    -------
    psf : np.ndarray, shape (kernel_size, kernel_size), float32
    """
    psf_name = '2D data/Microtubule/PSF/psf_emLambda525_dxy0.0313_NA1.3.tif'
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(psf_name) as f:
            psf_raw = tifffile.imread(f).astype(np.float64)

    # Center crop
    cy, cx = psf_raw.shape[0] // 2, psf_raw.shape[1] // 2
    half = kernel_size // 2
    psf = psf_raw[cy - half:cy + half, cx - half:cx + half]

    # Normalise
    psf = np.maximum(psf, 0.0)
    psf /= psf.sum()
    return psf.astype(np.float32)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: download
    download_data()

    # Step 2: extract MT images
    print('Extracting Microtubule images...')
    images = extract_mt_images(CACHE_ZIP)
    print(f'  Stack shape: {images.shape}, dtype: {images.dtype}')
    print(f'  Intensity range: [{images.min():.0f}, {images.max():.0f}], '
          f'mean={images.mean():.1f}')

    # Step 3: extract and save PSF
    print('Extracting PSF...')
    psf = extract_psf(CACHE_ZIP, kernel_size=32)
    print(f'  PSF shape: {psf.shape}, sum={psf.sum():.6f}')

    # Step 4: estimate noise parameters from a single frame
    from src.physics_model import estimate_noise_params
    beta1, beta2 = estimate_noise_params(
        images[0].astype(np.float64), bg=100.0)
    print(f'  Estimated noise params: beta1={beta1:.3f}, beta2={beta2:.3f}')

    # Step 5: save npz files
    raw_path = os.path.join(DATA_DIR, 'raw_data.npz')
    gt_path = os.path.join(DATA_DIR, 'ground_truth.npz')
    np.savez_compressed(raw_path, measurements=images)
    # ground_truth contains the PSF kernel (used for Stage 2 deconvolution evaluation)
    np.savez_compressed(gt_path, psf=psf[np.newaxis])   # shape (1, 32, 32)
    print(f'Saved: {raw_path}')
    print(f'Saved: {gt_path}')

    # Step 6: save metadata
    meta = {
        "modality": "TIRF fluorescence microscopy",
        "structure": "Microtubule (filamentous)",
        "n_frames": int(images.shape[0]),
        "image_height": int(images.shape[1]),
        "image_width": int(images.shape[2]),
        "pixel_size_um": 0.0313,
        "excitation_wavelength_nm": 488,
        "emission_wavelength_nm": 525,
        "na": 1.3,
        "camera": "sCMOS (Hamamatsu Orca Flash 4.0 v3)",
        "background_adu": 100,
        "avg_signal_above_bg_adu": float(np.median(
            images.mean(axis=(1, 2)) - 100)),
        "noise_model": {
            "description": "Poisson-Gaussian: sigma^2 = beta1*H(y-bg) + beta2",
            "beta1_estimated": round(float(beta1), 4),
            "beta2_estimated": round(float(beta2), 4),
            "beta1_theoretical_optimal": 1.0,
            "alpha_recorruption": 1.5,
            "filter_size": 5
        },
        "psf": {
            "file": "psf_emLambda525_dxy0.0313_NA1.3.tif",
            "pixel_size_um": 0.0313,
            "kernel_size": 32,
            "description": "2D PSF for Microtubule channel (525 nm emission, NA 1.3)"
        },
        "source": "Zenodo DOI: 10.5281/zenodo.7261163",
        "paper": "Qiao et al., Nature Communications 15, 4180 (2024)"
    }
    meta_path = os.path.join(DATA_DIR, 'meta_data.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved: {meta_path}')


if __name__ == '__main__':
    main()
