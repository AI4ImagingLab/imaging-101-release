# Reflection-Mode Optical Diffraction Tomography (rMS-FPT)

> Reconstruct the 3D refractive index of a sample on a reflective substrate from multi-angle reflected-light measurements using multi-slice Fourier ptychographic tomography.

> Domain: Biology | Keywords: tomography, wave imaging, phase retrieval | Difficulty: Hard

---

## Background

This task implements reflection-mode multi-slice Fourier ptychographic tomography (rMS-FPT) for 3D refractive index reconstruction of samples on reflective substrates.

**Reference**: Zhu et al., "Reflection-mode multi-slice Fourier ptychographic tomography", arXiv:2503.12246 (2025)

**GitHub**: https://github.com/bu-cisl/rMSBP-python

## Problem Description

Reconstruct the 3D refractive index (RI) distribution Δn(x,y,z) from intensity measurements under multiple illumination angles. The sample is placed on a reflective substrate (mirror), and light propagates through the sample, reflects, and propagates back before detection.

### Physics

- **Forward model**: Reflection-mode multi-slice beam propagation method (rMSBP)
- **Geometry**: Sample on mirror substrate, illumination from above, with epi-detection after round-trip propagation
- **Measurements**: Intensity images at multiple illumination angles from 4 illumination rings (56 angles total)
- **Sample**: 4-layer phantom with USAF-like patterns, each layer has RI 1.47 in a background of
    1.54(Δn = -0.07)

### Illumination / Incident Angles

The incident field is a tilted plane wave. Illumination angles are organised
as rings in NA-space, where each ring has a fixed illumination NA and multiple
azimuthal angles:

- Bright-field illumination (`type="BF"`): `NA_illu < NA_obj`, so the direct
  reflected/unscattered field remains inside the objective pupil and is
  recorded together with the scattered field
- Dark-field illumination (`type="DF"`): `NA_illu > NA_obj`, so the direct
  reflected/unscattered field falls outside the objective pupil and is rejected;
  the detected signal is therefore dominated by scattered light that is shifted
  back into the pupil

Current metadata uses:

- bright-field illumination: `NA=0.130` with 8 angles
- bright-field illumination: `NA=0.251` with 16 angles
- dark-field illumination: `NA=0.420` with 16 angles
- dark-field illumination: `NA=0.560` with 16 angles

### Sample Geometry: Discrete Layers with Gaps

 The validation phantom consists of four thin patterned scattering layers above a mirror, with each 500-nm-thick layer separated by a 10-μm homogeneous gap. The forward model remains a reflection-aware split-step BPM(standard split-step BPM: propagate → scatter per slice): the field propagates downward through the layered structure, reflects at the mirror, and then propagates upward through the same slices in reverse order. For this discrete-layer phantom, free-space propagation is explicitly applied across the inter-layer gaps.


```
             illumination →
Layer 0 ──gap── Layer 1 ──gap── Layer 2 ──gap── Layer 3 ──gap── Mirror
 0.5μm   10μm    0.5μm   10μm    0.5μm   10μm    0.5μm   10μm
                                                                        │ perfect mirror
                                                                        │ reflect field
            ← illumination
Layer 0 ──gap── Layer 1 ──gap── Layer 2 ──gap── Layer 3 ──gap── Mirror
 0.5μm   10μm    0.5μm   10μm    0.5μm   10μm    0.5μm   10μm

```

This leads to a **three-step BPM** per layer in the forward pass:

| Step | Operation | Physical meaning |
|------|-----------|-----------------|
| `propagate(dz_layer)` | Free-space propagation | Traverse the layer thickness |
| `scatter(Δn, dz_layer)` | Phase accumulation | RI contrast within the layer |
| `propagate(dz_gap)` | Free-space propagation | Traverse gap to the next layer |

The gap distance (`dz_gap=10μm`) is 20× the layer thickness (`dz_layer=0.5μm`), reflecting the physical scenario of thin etched patterns on spacer layers. If `dz_gap=0`, this reduces to the standard split-step BPM. In reflection geometry, light traverses every gap twice (down and back up), so the gap propagation has a significant cumulative effect on the measured intensity.

## Data Description

### Synthetic Phantom

- **Volume**: 4 layers × 128×128 pixels
- **Voxel size**: 0.274 × 0.274 × 1.0 μm (dy, dx, dz)
- **Background RI**: n0 = 1.54
- **RI contrast**: Δn = n(x,y,z) − n₀ = −0.07 (structures darker than background)
- **Patterns**: Layer 0 (vertical bars), Layer 1 (horizontal bars), Layer 2 (diagonal bars), Layer 3 (concentric rings)

### `data/raw_data.npz`

Simulated intensity measurements for all illumination angles.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `measurements` | `(1, 56, 128, 128)` | float64 | Intensity images — batch × n_angles × Ny × Nx |

### `data/ground_truth.npz`

Ground truth refractive index contrast volume.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `delta_n` | `(1, 4, 128, 128)` | float64 | RI contrast Δn = n(x,y,z) − n₀ — batch × Nz × Ny × Nx |

### `data/meta_data.json`

```json
{
    "volume_shape": [4, 128, 128],
    "res_um": [0.274, 0.274, 1.0],
    "wavelength_um": 0.632,
    "n0": 1.54,
    "NA_obj": 0.28,
    "illumination_rings": [
        {"NA": 0.130, "n_angles": 8,  "type": "BF"},
        {"NA": 0.251, "n_angles": 16, "type": "BF"},
        {"NA": 0.420, "n_angles": 16, "type": "DF"},
        {"NA": 0.560, "n_angles": 16, "type": "DF"}
    ],
    "dz_layer": 0.5,
    "dz_gap": 10.0,
    "ri_contrast": -0.07
}
```

## Method Hints

- **Inverse solver**: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
- **Regularization**: 2D total variation (per-slice)
- **Loss**: Amplitude MSE (||√I_pred - √I_meas||²)

## References
