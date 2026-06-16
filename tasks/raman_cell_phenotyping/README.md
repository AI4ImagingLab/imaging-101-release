# Raman Cell Phenotyping

> Unmix volumetric Raman spectroscopic data of THP-1 cells into biomolecular endmember spectra (DNA, protein, lipids) and their spatial abundance maps, using hyperspectral unmixing under a linear mixing model.

> Domain: chemistry | Keywords: Raman spectroscopy, hyperspectral unmixing | Difficulty: Medium

## Background

Raman spectroscopy is a label-free, non-destructive technique that probes molecular vibrations via inelastic light scattering. Each molecule produces a characteristic Raman spectrum (a "fingerprint"), so the observed spectrum at any point in a biological sample is a superposition of contributions from all constituent biomolecules. By acquiring spectra on a spatial grid and stacking multiple focal planes, one obtains a four-dimensional hyperspectral volume (x, y, z, wavenumber).

Extracting the individual molecular components and mapping their spatial distribution within a cell is the central task. This is a classic **hyperspectral unmixing** problem: given mixed observations, recover the pure component spectra (endmembers) and their fractional abundances at each voxel.

## Problem Description

The measurement at each voxel is modelled as a **linear mixture** of K unknown endmember spectra:

$$y_i = \sum_{k=1}^{K} a_{ik}\, e_k + n_i$$

where $y_i \in \mathbb{R}^B$ is the observed spectrum at voxel $i$, $e_k \in \mathbb{R}^B$ are endmember spectra, $a_{ik} \geq 0$ are fractional abundances with $\sum_k a_{ik} = 1$, and $n_i$ is additive noise.

In matrix form: $Y = A E + N$, where $Y$ is $(N_\text{pixels} \times B)$, $A$ is $(N_\text{pixels} \times K)$, and $E$ is $(K \times B)$.

The inverse problem is **bilinear** — both $A$ and $E$ are unknown — and **ill-posed** because:
1. The number of endmembers $K$ must be chosen or estimated.
2. Noise, baseline fluorescence, and cosmic-ray artefacts contaminate raw spectra.
3. The non-negativity and sum-to-one constraints on $A$ restrict the feasible set.

**Input**: a preprocessed spectral volume of shape (40, 40, 10, B'), where B' is the number of spectral channels in the fingerprint region (700–1800 cm$^{-1}$).

**Output**: K endmember spectra of shape (K, B') and K abundance maps of shape (40, 40, 10).

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `spectral_volume` | (1, 40, 40, 10, 1600) | float32 | Raw Raman intensity volume [counts] |
| `spectral_axis` | (1600,) | float64 | Wavenumber axis [cm$^{-1}$] |

The volume contains 10 z-layers of 40×40 spatial pixels, each with 1600 spectral channels spanning approximately 0–3700 cm$^{-1}$.

### data/meta_data.json

| Key | Value | Description |
|-----|-------|-------------|
| `spatial_shape` | [40, 40, 10] | (X, Y, Z) voxel grid |
| `n_spectral_points` | 1600 | Channels before cropping |
| `fingerprint_region_cm1` | [700, 1800] | Region of interest [cm$^{-1}$] |
| `laser_excitation_nm` | 532 | Excitation wavelength [nm] |
| `cell_type` | "THP-1" | Human monocytic cell line |
| `n_endmembers` | 5 | Target number of endmembers |

### Ground truth

No `ground_truth.npz` is provided for this task; the data is from real Raman measurements of THP-1 cells where the true endmember spectra and abundance maps are not independently known. Evaluation uses a `baseline_reference.npz` containing the reference N-FINDR + FCLS unmixing result.

### data/baseline_reference.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `abundance_lipids` | (1, 40, 40) | float64 | Reference lipid abundance, layer 5 |
| `abundance_nucleus` | (1, 40, 40) | float64 | Reference nucleus abundance, layer 5 |
| `abundance_cytoplasm` | (1, 40, 40) | float64 | Reference cytoplasm abundance, layer 5 |
| `abundance_background` | (1, 40, 40) | float64 | Reference background abundance, layer 5 |

## Method Hints

The task decomposes into two stages:

1. **Preprocessing**: raw Raman spectra require spectral cropping to the fingerprint region, cosmic-ray removal, spectral smoothing, baseline correction (to remove fluorescence background), and intensity normalisation. These are standard steps in Raman data analysis.

2. **Hyperspectral unmixing**: a geometry-based endmember extraction algorithm (N-FINDR) identifies pure spectral signatures by finding the simplex of maximum volume in the spectral point cloud. Given the endmembers, per-voxel abundances are obtained via constrained least squares with non-negativity and sum-to-one constraints (FCLS).

The key insight is that N-FINDR is a convex-geometry method — it assumes that at least one voxel per endmember contains a nearly pure spectrum. This is typically satisfied in cell imaging where the nucleus, cytoplasm, lipid droplets, and surrounding medium occupy distinct regions.

## References

- Kallepitis, C., Bergholt, M., Mazo, M. et al. Quantitative volumetric Raman imaging of three dimensional cell cultures. *Nat Commun* 8, 14843 (2017).
- Winter, M.E. N-FINDR: An algorithm for fast autonomous spectral end-member determination in hyperspectral data. *Imaging Spectrometry V*, SPIE, 1999.
- Heinz, D.C. Fully constrained least squares linear spectral mixture analysis method for material quantification in hyperspectral imagery. *IEEE Trans. Geosci. Remote Sens.* 39(3), 529–545 (2001).
