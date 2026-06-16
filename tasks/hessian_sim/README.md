# Hessian Structured Illumination Microscopy

> One-sentence summary: reconstruct a 2x super-resolved, denoised fluorescence image from six phase-shifted 2-beam SIM raw frames per time point, then stabilise the time series with a Hessian prior.

> Domain: biology / super-resolution microscopy | Keywords: structured illumination microscopy, super-resolution, regularised inverse problem | Difficulty: Medium

## Background

Fluorescence microscopy is the workhorse of modern cell biology because it
maps specific biomolecules with high chemical specificity, but its spatial
resolution is fundamentally limited by diffraction: two point sources closer
than roughly half the emission wavelength cannot be distinguished through a
wide-field objective. Structured Illumination Microscopy (SIM) overcomes this
bound in a computational way — the sample is illuminated with a known periodic
pattern so that high spatial-frequency content is aliased down into the
passband of the objective, and several raw frames acquired with shifted /
rotated patterns are then jointly deconvolved to recover a 2x super-resolved
image.

SIM is attractive for live-cell imaging because it uses ordinary fluorescent
labels, works at moderate light doses, and can run at video rate. It has become
the method of choice for following fast sub-cellular dynamics such as
cytoskeletal remodelling, mitochondrial fission, and endocytic trafficking.

The price for this speed is noise. Each super-resolved frame is assembled from
only a handful of very short exposures, so the reconstruction is dominated by
shot noise and by the amplification the inverse filter applies to weakly
sampled Fourier components. Naïve Wiener-style SIM reconstructions therefore
look grainy, especially at the edges of the extended frequency support where
the signal-to-noise ratio is lowest. A structural prior that suppresses this
noise while preserving the genuinely high-resolution features is what makes a
SIM reconstruction usable for biological analysis.

## Problem Description

We model a 2-beam SIM acquisition of a 2D fluorophore density ``x(r)``. For
each of $n\_\mathrm{ang}$ illumination angles indexed by ``a`` and each of
$n_\mathrm{ph}$ phase steps indexed by ``p``, the measurement is

```math
y_{a,p}(r) = \bigl[\,(1 + m\cos(2\pi k_a\!\cdot\! r + \varphi_{a,p}))\,x(r)\,\bigr] * h(r) + n_{a,p}(r),
```

where $k_a$ is the pattern wavevector, $\varphi_{a,p}$ is the phase of
step ``p``, $m \le 1$ is the pattern modulation depth, $h(r)$ is the
microscope point spread function (whose Fourier transform is the OTF
``H(k)``), and $n_{a,p}$ is shot noise. The cosine modulation folds three
frequency components of ``x`` into each raw frame:

```math
\tilde y_{a,p}(k) = H(k)\Bigl[\,\tilde x(k) + \tfrac{m}{2}e^{i\varphi_{a,p}}\tilde x(k - k_a) + \tfrac{m}{2}e^{-i\varphi_{a,p}}\tilde x(k + k_a)\Bigr] + \tilde n_{a,p}(k).
```

The inverse problem has three coupled pieces of missing information:

1. the pattern parameters $\lbrace k_a, \varphi_{a,p}, m\rbrace$ (calibration),
2. the super-resolved object ``x`` (band separation + Wiener inversion),
3. noise suppression that respects the unusually anisotropic frequency
   support of the reconstructed Fourier spectrum.

Steps (1)–(2) alone are ill-conditioned near the edge of the extended OTF
support, and the residual reconstruction noise couples strongly across the
time axis. This motivates a regularised denoising step (3) that acts jointly
in x, y and t using second-order (Hessian) derivatives, preserving the
piecewise-smooth biological structure while removing the streak-like noise
pattern characteristic of Wiener SIM.

**Inputs.** ``(12, 6, 256, 256)`` block of raw frames — 12 time points, 6
frames per time point (2 angles × 3 phases), plus imaging parameters.
**Output.** ``(12, 512, 512)`` stack of super-resolved, denoised frames.

## Data Description

### `data/raw_data.npz`

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `raw_frames` | `(1, 72, 256, 256)` | float32 | Simulated 2-beam SIM raw stack, ordered as ``[a_0 p_0, a_0 p_1, a_0 p_2, a_1 p_0, a_1 p_1, a_1 p_2]`` for each of 12 time points. Intensity in detector counts. |

No `ground_truth.npz` is provided for this task; the reference reconstruction
in `evaluation/reference_outputs/wiener_sim.npz` (the Wiener-SIM output of
the ported MATLAB pipeline) is used as the evaluation target for NCC/NRMSE
comparison, since the original supplementary simulation does not ship a
noise-free ground-truth image.

### `data/meta_data.json`

| Field | Description |
|---|---|
| `nangles` | number of illumination angles (2) |
| `nphases` | number of phase steps per angle (3) |
| `wavelength_nm` | excitation / emission wavelength in nm (488) |
| `numerical_aperture` | objective NA used for the synthetic OTF (0.9) |
| `pixel_size_nm` | camera pixel size at the sample in nm (65) |
| `spjg` | integer phase-step ratios used to build the band-separation matrix |
| `synthetic_otf` | `true` — the OTF is regenerated from NA/wavelength/pixel_size |
| `frames_per_timepoint` | `nangles * nphases` = 6 |

## Method Hints

The Hessian-SIM reference implementation is a six-stage sequential pipeline.

1. **Pattern frequency estimation.** Band-separate the raw frames with a
   known phase matrix, normalise amplitude so only the phase survives, and
   cross-correlate the 0th-order and first-order bands to locate each
   illumination wavevector $k_a$. Refine sub-pixel via a greedy
   coordinate-descent merit search.
2. **Modulation depth + phase offset.** Build two OTF overlap regions (a wide
   one for phase, a narrow one for depth). For each sideband order, form a
   complex modulation map $\mathrm{cm} = \mathrm{cross}/\mathrm{overlap}$
   and histogram its angle (phase) and magnitude (depth). EMD-smoothed peak
   finding returns $\varphi_{a,p}$ and ``m``.
3. **Wiener-SIM reconstruction.** Unmix bands with a phase matrix rebuilt
   from the estimated $\varphi$, shift each band to its true frequency
   location via the Fourier shift theorem, weight by the estimated modulation
   depth, and recombine with a generalised Wiener filter

$$
   \hat x(k) = \frac{\sum_{a,p} w_{a,p} H^{*}_{a,p}(k)\,\tilde y_{a,p}(k)}{\sum_{a,p}|H_{a,p}(k)|^2 + \varepsilon}
$$

4. **Hessian denoising.** Solve the split-Bregman problem

$$
   \min_x  \lambda \sum_{k\in\mathcal H} \lVert\Phi_k x\rVert_1 + \frac{\mu}{2}\lVert x - \hat x_{\text{Wiener}}\rVert_2^2,
$$

   where $\mathcal H = \lbrace\partial_{xx},\partial_{yy},\partial_{zz},\partial_{xy},\partial_{xz},\partial_{yz}\rbrace$ is the six-term second-order Hessian basis. The L1 on the Hessian favours piecewise-smooth (rather than piecewise-constant) structures, and the joint xyt treatment propagates noise suppression across time.

5. **TV denoising** (baseline comparison). Same split-Bregman with the four
   first-order differences instead of the Hessian basis — reveals the
   staircase artefacts that motivate the Hessian prior.

6. **Running average.** Three-frame temporal boxcar over the denoised stack
   to smooth residual flicker.

Keep the coordinate conventions in `src/physics_model.py` consistent
throughout — the Fourier shift, pattern wavevector, and pixel grids must all
use the same centred-FFT convention or the reconstructed Fourier bands will
be positioned incorrectly.

## References

1. Huang, X. *et al.* Fast, long-term, super-resolution imaging with Hessian
   structured illumination microscopy. *Nature Biotechnology* **36**, 451–459
   (2018).
2. Heintzmann, R. & Cremer, C. Laterally modulated excitation microscopy:
   improvement of resolution by using a diffraction grating. *Proc. SPIE*
   **3568**, 185–196 (1999).
3. Goldstein, T. & Osher, S. The split Bregman method for L1-regularized
   problems. *SIAM J. Imaging Sci.* **2**, 323–343 (2009).
