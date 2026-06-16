# Conventional Ptychography Reconstruction

> Recover the complex amplitude and phase of a sample from overlapping far-field diffraction patterns using the momentum-accelerated PIE (mPIE) algorithm.

> Domain: Physics | Keywords: phase retrieval, ptychography | Difficulty: Medium

---

## Background

Conventional ptychography (CP) is a lensless computational imaging technique that
reconstructs a complex-valued sample (amplitude and phase) from a set of far-field
diffraction patterns. It is widely used in synchrotron x-ray microscopy, electron
ptychography, and tabletop EUV microscopy.

This task implements the full CP reconstruction pipeline via a self-contained mPIE
solver, reproducing a synthetic data example inspired by the PtyLab benchmark
(Loetgering et al., Opt. Express 2023).

## Problem Description

```
  [source] → pinhole → lens → [probe P(r)] → [object O(r)] → [detector I(q)]
                                                    ↑
                              scanned by Fermat spiral (100 positions)
```

| Parameter | Value |
|---|---|
| Wavelength | 632.8 nm |
| Sample–detector distance z_o | 50 mm |
| Detector size | 128 × 128 px |
| Effective detector pixel size | 72 μm |
| Probe pixel size | ~3.433 μm |
| Object field of view | ~1.86 mm |
| Scan positions | 100 (Fermat spiral) |

## Data Description

### `data/raw_data.npz`

Far-field diffraction patterns from a synthetic ptychography experiment (100 Fermat-spiral scan positions).

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `ptychogram` | (100, 128, 128) | float32 | Far-field diffraction intensities (one frame per scan position) |
| `encoder` | (100, 2) | float32 | Scan positions in meters [row, col] |

### `data/ground_truth.npz`

Ground-truth complex transmission function of the simulated sample.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `object` | (542, 542) | complex64 | Ground-truth complex transmission function O(r) |

### `data/meta_data.json`

JSON file with optical system parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `wavelength_m` | 6.328e-7 | He-Ne laser wavelength (m) |
| `zo_m` | 0.05 | Sample–detector distance (m) |
| `Nd` | 128 | Detector size (pixels) |
| `dxd_m` | 7.2e-5 | Effective detector pixel size (m) |
| `No` | 542 | Object array size (pixels) |
| `dxp_m` | 3.433e-6 | Probe/object pixel size (m) |
| `entrance_pupil_diameter_m` | 2.644e-4 | Beam diameter at sample plane (m) |
| `num_scan_positions` | 100 | Number of Fermat-spiral scan positions |
| `propagator` | "Fraunhofer" | Far-field diffraction model |

## Method Hints

The key insight in ptychography is that **overlap** between adjacent scan positions
provides the redundancy needed to solve the phase retrieval problem. The
exit-wave estimate is updated at each scan position using the ePIE formula:

    O_{n+1} = O_n + β · P_n* · (ψ̂_j − ψ_j) / max|P_n|²

where ψ̂_j is obtained by projecting the measured intensity onto the current
detector wave estimate.

The mPIE momentum update prevents slow convergence on large objects by adding
a global gradient accumulation step after every T iterations.

## References

L. Loetgering, M. Du, D. B. Flaes, T. Aidukas, F. Wechsler, D. S. Penagos Molina,
M. Rose, A. Pelekanidis, W. Eschen, J. Hess, T. Wilhein, R. Heintzmann, J. Rothhardt,
and S. Witte, "PtyLab.m/py/jl: a cross-platform, open-source inverse modeling toolbox
for conventional and Fourier ptychography," Opt. Express 31, 13763–13797 (2023).
