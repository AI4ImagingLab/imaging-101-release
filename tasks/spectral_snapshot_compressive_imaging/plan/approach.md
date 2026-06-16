# Approach: PnP-CASSI

## Problem

Spectral snapshot compressive imaging (SCI) captures a 3D hyperspectral data cube (x, y, wavelength) in a single 2D measurement using coded apertures. The inverse problem is to reconstruct the full spectral cube from this compressed measurement.

In a coded aperture snapshot spectral imager (CASSI), a physical mask modulates the scene, and a dispersive element (prism) shifts each spectral channel by a wavelength-dependent amount before they are summed onto the detector. The forward model is:

    y = sum_i (Phi_i * S_i(x_i))

where Phi_i is the mask for channel i, S_i is the spectral shift operator, and x_i is the i-th spectral band.

## Method: Plug-and-Play (PnP) Framework

We use an optimization-based reconstruction framework where the denoiser acts as an implicit prior:

### GAP (Generalized Alternating Projection)

The GAP algorithm alternates between:
1. **Data fidelity step**: Project onto the measurement constraint
   - x <- x + lambda * At((y - A(x)) / Phi_sum)
   - With acceleration: maintain a running residual y1 for faster convergence
2. **Prior step**: Apply a denoiser to regularize the estimate
   - Early iterations (k < 83): TV denoising (fast, coarse regularization)
   - Later iterations (k >= 83): Deep CNN denoiser (HSI-SDeCNN) for fine detail recovery

### Deep Denoiser: HSI-SDeCNN

The denoiser operates on a sliding spectral window of 7 bands centered at each target band. Key design:
- PixelUnshuffle downsampling (factor 2) to reduce spatial resolution
- 15-layer CNN with 128 channels
- PixelShuffle upsampling to restore resolution
- Noise level map concatenated as an extra channel (flexible noise handling)
- Boundary bands use mirrored padding to maintain the 7-band input window

### Hybrid Denoising Strategy

The algorithm uses a staged approach:
- **Iterations 1-82**: TV denoising only (establishes coarse structure)
- **Iterations 83+**: Alternating between CNN (3 iterations) and TV (1 iteration)
- This hybrid approach stabilizes convergence while leveraging the CNN's detail recovery

### ADMM Variant

An ADMM-based solver is also provided as an alternative, using the same deep denoiser but with an augmented Lagrangian framework and dual variable updates.

## References

Zheng et al., "Deep plug-and-play priors for spectral snapshot compressive imaging," Photon. Res. 9, B18-B29 (2021)

## Default Parameters

These values are hard-coded in `main.py` and were removed from `meta_data.json` (which is reserved for imaging parameters only):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lambda` | 1 | GAP regularization parameter λ |
| `accelerate` | True | Use Nesterov acceleration |
| `iter_max` | 20 | GAP outer iterations |
| `tv_iter_max` | 5 | TV denoiser inner iterations per GAP step |
| `sigma` | [130]×8 | Denoiser noise level schedule (one per stage) |
