# Plane Wave Ultrasound Imaging (Stolt's f-k Migration)

> Reconstruct focused B-mode ultrasound images from multi-angle plane-wave RF data using frequency–wavenumber domain migration, achieving O(N log N) complexity versus O(N²) for delay-and-sum beamforming.

> Domain: Medicine | Keywords: wave imaging, ultrafast imaging | Difficulty: Medium

## Background

Conventional ultrasound imaging fires focused beams sequentially, yielding frame rates of tens of images per second. **Ultrafast imaging** transmits a single unfocused plane wave that insonifies the entire medium simultaneously, achieving frame rates of thousands per second. The trade-off is that a single plane-wave acquisition produces low-contrast images; high-quality images require coherent compounding of several steered plane-wave firings.

The standard reconstruction algorithm for focused-beam ultrasound is delay-and-sum (DAS) beamforming, which is an O(N²) operation. For plane-wave imaging, **Stolt's f-k migration** — borrowed from seismic reflection imaging — reduces this to O(N log N) via FFTs, achieving over 100× speedup with equivalent image quality.

The key physical model is the **exploding reflector model (ERM)**: instead of tracing two-way travel paths (transmit + receive), the scatterers are imagined to "explode" at t = 0 and propagate as a half-velocity wave (v = c/2). Under this model, the round-trip wavefield is equivalent to a one-way wavefield with an effective medium velocity v. For a steered plane wave at angle θ, the ERM velocity is

$$v = \frac{c}{\sqrt{1 + \cos\theta + \sin^2\theta}}$$

Stolt migration then maps the recorded (t, x) data to a focused (z, x) image using an exact analytical dispersion relation in the frequency–wavenumber (f–k) domain.

## Problem Description

Given RF (radio-frequency) backscattered signals recorded by a 128-element linear transducer array after transmitting plane waves at 7 steering angles (−1.5° to +1.5° in 0.5° steps), reconstruct a focused B-mode ultrasound image of the tissue phantom.

**Input:** Raw RF signals of shape (N_t, N_x, N_angles) — time samples × array elements × steering angles.

**Output:** A focused B-mode image of shape (N_t, N_x), with sufficient lateral resolution to resolve individual wire targets (≲ 1 mm FWHM) and sufficient contrast to distinguish circular cyst inclusions from surrounding tissue.

The challenge is that a single plane-wave transmit produces an unfocused wavefield: each point scatterer contributes a characteristic hyperbolic arc across all receive elements and time samples. Refocusing these arcs into sharp point-spread functions is the core of the reconstruction problem.

## Data Description

Two real ultrasound datasets are included, both acquired with a 128-element linear transducer (SonixRP, Ultrasonix Medical), sampling frequency 20 MHz, element pitch 298 µm, speed of sound 1540 m/s, 7 compounding angles (−1.5° to +1.5° in 0.5° steps).

### `data/raw_data.npz`

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `RF_fibers` | (1, 2688, 128, 7) | float32 | RF signals in raw 8-bit ADC counts, nylon-fiber (point-target) phantom. Axes: batch, time samples, array elements, steering angles. DC removal is applied later in preprocessing. |
| `RF_cysts`  | (1, 1536, 128, 7) | float32 | RF signals in raw 8-bit ADC counts, circular-cyst phantom. Same axis order; acquisition starts at t₀ = 5×10⁻⁵ s after the first transmit. |

No `ground_truth.npz` is provided for this task; image quality is evaluated via lateral resolution (FWHM of point-spread function at each wire target) and contrast-to-noise ratio (CNR) of each cyst. `data/baseline_reference.npz` contains the reference Stolt f-k B-mode images for NCC/NRMSE evaluation.

### `data/baseline_reference.npz`

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `bmode_fibers` | (1, 2688, 128) | float32 | Reference B-mode image, nylon-fiber phantom (Stolt f-k, 7 angles compounded), arbitrary display units |
| `bmode_cysts`  | (1, 1536, 128) | float32 | Reference B-mode image, circular-cyst phantom (Stolt f-k, 7 angles compounded), arbitrary display units |

### `data/meta_data.json`

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `dataset_1` | object | JSON object | Fibers acquisition metadata: `description`, `c` (m/s), `fs` (Hz), `pitch` (m), `TXangle_deg` (degrees), `TXangle_rad` (radians), `t0` (s), `n_elements`, `n_samples`, `n_angles` |
| `dataset_2` | object | JSON object | Cysts acquisition metadata with the same fields and units as `dataset_1`; here `t0 = 5×10⁻⁵ s` |

## Method Hints

**Algorithm family:** Stolt f-k (frequency–wavenumber) migration.

**Key decomposition:** The round-trip RF wavefield is decomposed by 2-D FFT into spatial frequency (k_x) and temporal frequency (f) components. Stolt migration performs an *exact* analytical coordinate change

$$f \;\longrightarrow\; f_{kz} = v\sqrt{k_x^2 + \frac{4 f^2}{\beta^2 c^2}}$$

implemented as a 1-D interpolation along the frequency axis for each k_x slice, followed by an obliquity-factor correction and a 2-D IFFT. The interpolation step is the computational core; higher-order interpolation (sinc) yields better accuracy than linear interpolation but is more expensive.

**Why it suits the problem:** Plane-wave transmits excite the full aperture simultaneously, so the received wavefield is a superposition of spherical waves that can be efficiently decomposed in the f-k domain. The O(N log N) complexity allows real-time imaging at thousands of frames per second.

**Steering compensation:** For a steered plane wave at angle θ, arrival-time shifts are removed in the frequency domain before spatial FFT (multiply by `exp(-2πi f t_shift(θ))`), and the f-k mapping uses the angle-dependent ERM velocity and β factor. After inverse axial FFT, a lateral phase ramp `exp(-2πi k_x Δx(z))` corrects for remaining steering-induced displacement.

## References

- Garcia D, Le Tarnec L, Muth S, Montagnon E, Porée J, Cloutier G. Stolt's f-k migration for plane wave ultrasound imaging. *IEEE Trans Ultrason Ferroelectr Freq Control.* 2013;60(9):1853-1867.
- Reference implementation: https://github.com/rehmanali1994/Plane_Wave_Ultrasound_Stolt_F-K_Migration.github.io
- Data source: https://www.biomecardio.com/en/index.html (PICMUS / Garcia lab datasets)
