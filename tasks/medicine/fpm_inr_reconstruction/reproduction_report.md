# FPM-INR Reproduction Report

**Paper:** FPM-INR: Fourier ptychographic microscopy image stack reconstruction using implicit neural representations
**Authors:** Haowen Zhou, Brandon Y. Feng, Haiyun Guo, Siyu (Steven) Lin, Mingshu Liang, Christopher A. Metzler, Changhuei Yang
**Published:** Optica, Vol. 10, No. 12, Dec 2023
**DOI:** https://doi.org/10.1364/OPTICA.505283

**Reproduction Date:** 2026-03-24
**Reproduced by:** schen47
**Hardware:** NVIDIA A40 GPU on gpub055

---

## Reproduced Result: Fig. 2 — Human Blood Smear Image Stack

### Target
Fig. 2 presents the 3D image stack reconstruction of a tilted human blood smear sample using FPM-INR. The key quantitative metric is the **L2 error (MSE) on the all-in-focus image** computed via the Normal Variance method, comparing FPM-INR output against brightfield ground truth.

**Paper-reported L2 error for FPM-INR: 1.41 × 10⁻³**

### Experimental Setup (matching paper Section 4.1)
- **Sample:** Human blood smear (BloodSmearTilt), tilted at ~4° to optical axis
- **Illumination:** 68 LEDs, center wavelength 522 nm (green channel)
- **Objective:** 10×/0.25 NA
- **ROI:** 1024 × 1024 pixels (raw), upsampled to 2048 × 2048
- **Z-range:** -20 μm to +20 μm, 161 z-slices for ground truth comparison

### Hyperparameters (matching paper)
| Parameter | Value |
|-----------|-------|
| Epochs | 15 |
| Learning rate | 1 × 10⁻³ |
| LR decay | 10× every 6 epochs |
| Feature channels (Q) | 32 |
| Feature plane resolution | 512 × 512 |
| Z-axis discretization | 5 planes |
| Optimizer | Adam |
| Loss function | Smooth L1 |
| torch.compile backend | inductor |
| Mixed precision (bfloat16) | Enabled |

### Z-slice Selection Strategy (matching paper Section 3.3)
- **Even epochs:** Uniform z-planes corresponding to feature discretization
- **Odd epochs:** Random z-planes with interpolation

### Results

#### Primary Metric: All-in-Focus L2 Error (Fig. 2 metric)

| Method | L2 Error (MSE) | Status |
|--------|---------------|--------|
| Paper FPM (baseline) | 2.34 × 10⁻³ | — |
| Paper FPM-INR | 1.41 × 10⁻³ | Target |
| **Our FPM-INR** | **1.33 × 10⁻³** | **Reproduced** |

Our reproduction achieves an L2 error of **1.325 × 10⁻³**, which is slightly better than the paper's reported 1.41 × 10⁻³. This difference is within expected variance from random initialization and hardware differences (paper used RTX A6000; we used A40).

The all-in-focus L2 error is computed as:
1. Generate 161 z-slices from the trained model (z = -20 to 20 μm)
2. Normalize amplitude images to [0, 1]
3. Compute all-in-focus image using Normal Variance method (patchsize=64, patchpace=16)
4. Compute MSE between mean-subtracted FPM-INR and GT all-in-focus images

#### Additional Per-Slice Metrics (supplementary, not in paper)

| Metric | Value |
|--------|-------|
| Per-slice PSNR (mean ± std) | 22.79 ± 0.55 dB |
| Per-slice SSIM (mean ± std) | 0.7753 ± 0.0341 |
| Best slice PSNR | 24.70 dB (z = -20.00 μm) |
| Worst slice PSNR | 21.84 dB (z = 10.00 μm) |

#### Training Performance
| Metric | Value |
|--------|-------|
| Training time (15 epochs) | ~7 minutes |
| Model inference (161 slices) | ~10 seconds |
| All-in-focus computation | ~114 seconds |
| Model size | 65 MB (saved weights) |

### Data Files
- **Input data:** `data/BloodSmearTilt/BloodSmearTilt_g.mat` (raw measurements)
- **Ground truth:** `data/BloodSmearTilt/BloodSmearTilt_g_GT.mat` (brightfield z-stack, 1024×1024×161)
- **Note:** Files are named `_g` per the original data source. The README erratum suggests renaming to `_r`, but the data content is green channel (522 nm). We used `--color g` to match filenames.

### Output Files

```
trained_models/
  BloodSmearTilt_g.pth              # Trained model weights (65 MB)

vis/feat32_3D_eval/
  allfocus_comparison.png           # GT vs FPM-INR all-in-focus + error map
  gt_comparison_metrics.png         # Per-slice L2, PSNR, SSIM curves
  gt_comparison_visual.png          # Side-by-side z-slice comparisons
  e_0.png                           # Amplitude & phase at epoch 0
  e_4.png                           # Amplitude & phase at epoch 4
  e_9.png                           # Amplitude & phase at epoch 9
  e_14.png                          # Amplitude & phase at epoch 14 (final)
```

### Reproduction Scripts

```
run_and_eval.py       # Training + per-slice GT evaluation
eval_allfocus.py      # All-in-focus evaluation (reproduces Fig. 2 metric)
eval_gt.py            # Per-slice evaluation (PSNR, SSIM, L2)
check_shapes.py       # Debug utility for model output shapes
```

### Environment

```
Node:        gpub055
GPU:         NVIDIA A40 (46 GB)
CUDA:        12.8
Python:      3.9.23
PyTorch:     2.0.1+cu118
Conda env:   /projects/bgkc/schen47/conda_envs/fpm_inr
```

### Conclusion

The key quantitative result from Fig. 2 of the paper — the all-in-focus L2 error of FPM-INR on the human blood smear sample — has been **successfully reproduced**. Our result (1.33 × 10⁻³) closely matches and slightly improves upon the paper's reported value (1.41 × 10⁻³).
