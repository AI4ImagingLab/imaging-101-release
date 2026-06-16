# Approach

## Problem Summary

Given a volumetric Raman spectroscopy dataset of THP-1 cells (40×40×10 voxels, 1600 spectral channels), recover 5 endmember spectra and their spatial abundance maps using a linear mixing model.

## Preprocessing Pipeline

Raw Raman spectra suffer from background fluorescence, cosmic-ray spikes, and detector noise. The preprocessing pipeline applies five sequential steps:

1. **Spectral cropping** to the fingerprint region (700–1800 cm⁻¹) — retains the most diagnostically informative bands while discarding noisy edges and the silent region.

2. **Cosmic-ray removal** (Whitaker-Hayes) — detects spikes via modified z-scores of spectral first differences; iteratively replaces flagged channels with the mean of clean neighbours. Parameters: `kernel_size=3`, `threshold=8`.

3. **Savitzky-Golay smoothing** — fits local polynomials in sliding windows to reduce high-frequency noise while preserving peak shapes. Parameters: `window_length=7`, `polyorder=3`.

4. **Asymmetric Least Squares (ASLS) baseline correction** — iteratively fits a smooth baseline that preferentially follows spectral valleys, removing broad fluorescence background. Parameters: `lam=1e6`, `p=0.01`.

5. **Global min-max normalisation** — scales all spectra to [0, 1] using a single global minimum and maximum across the entire volume (`pixelwise=False`).

## Unmixing Algorithm

### Endmember Extraction: N-FINDR

N-FINDR (Winter 1999) extracts K endmembers by finding the set of K pixels whose spectra form the simplex of maximum volume in spectral space:

1. Reduce spectral dimensionality from B to K-1 via PCA.
2. Initialise K random pixel spectra as candidate endmembers.
3. For each pixel in the dataset, test whether replacing each current endmember with that pixel's spectrum increases the simplex volume (computed as |det(M)| of the augmented endmember matrix).
4. Iterate until no replacements increase the volume.

The algorithm assumes at least one pixel per endmember is spectrally pure (or nearly so).

### Abundance Estimation: FCLS

Given K endmembers E = (e₁, ..., eₖ), Fully-Constrained Least Squares estimates per-pixel abundances by solving:

    min_a ||y - E^T a||²
    s.t.  a ≥ 0,  1^T a = 1

This is a convex quadratic program per pixel. The sum-to-one constraint ensures physical interpretability of abundances as fractional compositions.

## Endmember Assignment

After unmixing, endmembers are identified by inspecting characteristic Raman peaks:
- **DNA/Nucleus**: 789 cm⁻¹ (cytosine ring breathing)
- **Protein**: 1008 cm⁻¹ (phenylalanine)
- **Lipids**: 1066, 1134, 1303, 1443, 1747 cm⁻¹ (C-C stretch, CH₂ deformation, C=O)
- **Cytoplasm**: broad protein + lipid features
- **Background**: featureless or substrate signal

## Expected Outputs

- 5 endmember spectra (K × B')
- 5 abundance maps (40 × 40 × 10 each)
- Merged false-colour reconstruction at selected z-layer
