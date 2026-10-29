# EHT Black Hole Closure Imaging

> Reconstruct a radio brightness image from sparse, gain-corrupted interferometric measurements using closure quantities that cancel station gains.
> Domain: Astronomy | Keywords: radio interferometry, closure imaging, inverse imaging | Difficulty: Hard

## Background

Very long baseline interferometry combines radio telescopes separated by continental distances to resolve structure far smaller than any single telescope can distinguish. Each telescope pair samples one spatial frequency of the sky brightness. Earth rotation changes the sampled frequencies over an observation.

At millimetre wavelengths, atmospheric and instrumental changes perturb each station's amplitude and phase. Closure phases and closure amplitudes combine several baselines so that station-based gains cancel. They enable imaging when direct visibility calibration is unreliable, including observations near a black hole's event horizon.

## Problem Description

Let $I_{pq}$ be the nonnegative brightness in Jy/pixel at image pixel $(p,q)$, and let $(u_b,v_b)$ be baseline $b$ in wavelengths. The noiseless visibility follows the discretized Fourier model


```math
V_b = P(u_b,v_b)\sum_{p,q} I_{pq}\exp\left[2\pi i(u_b l_{pq}+v_b m_{pq})\right],
```


where $(l_{pq},m_{pq})$ are angular pixel coordinates in radians. The triangular pixel response is $P(u,v)=\mathrm{sinc}^2(u\Delta)\mathrm{sinc}^2(v\Delta)$, with $\mathrm{sinc}(t)=\sin(\pi t)/(\pi t)$ and pixel pitch $\Delta$ in radians. The actual visibility on stations $i,j$ is $Y_{ij}=g_i g_j^*V_{ij}+\epsilon_{ij}$, where $g_i$ is a complex station gain and $\epsilon_{ij}$ is measurement noise. For a station triangle, the closure phase is $\arg(Y_{ij}Y_{jk}Y_{ki})$; for a quadrangle, the log closure amplitude is $\log|Y_{ij}|+\log|Y_{kl}|-\log|Y_{ik}|-\log|Y_{jl}|$. Both cancel ideal station gains. Sparse frequency coverage and lost absolute position make the reconstruction underdetermined.

Save `output/reconstruction.npz` relative to the task workspace. It must contain the required key `reconstruction`: a finite, real `float64` array of shape `(1, 64, 64)` in Jy/pixel. Axes are `(batch, image row, image column)`; rows and columns follow the Fourier pixel grid described below. Preserve the physical flux scale, with total source flux approximately 0.6 Jy. The evaluator compares the declared array directly with a hidden reference using cosine NCC (higher is better) and dynamic-range-normalized RMS error (lower is better). It applies no crop, translation, phase, or flux alignment.

## Data Description

`data/raw_data.npz` contains one observation batch. Closure quantities and their UV legs come from the same gain-corrupted acquisition; no calibrated visibilities are supplied.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `vis_corrupt` | `(1, 421)` | complex128 | Measured gain-corrupted visibility, Jy. |
| `uv_coords` | `(1, 421, 2)` | float64 | Visibility $(u,v)$ coordinates, wavelengths. |
| `sigma_vis` | `(1, 421)` | float64 | Baseline thermal-noise standard deviation, Jy. |
| `station_ids` | `(1, 421, 2)` | int64 | Zero-based station pair indices, dimensionless. |
| `cp_corrupt_values_deg` | `(1, 269)` | float64 | Measured closure phase, degrees. |
| `cp_corrupt_sigmas_deg` | `(1, 269)` | float64 | Closure-phase uncertainty, degrees. |
| `cp_u1` | `(1, 269, 2)` | float64 | First directed triangle baseline $(u,v)$, wavelengths. |
| `cp_u2` | `(1, 269, 2)` | float64 | Second directed triangle baseline $(u,v)$, wavelengths. |
| `cp_u3` | `(1, 269, 2)` | float64 | Third directed triangle baseline $(u,v)$, wavelengths. |
| `lca_corrupt_values` | `(1, 233)` | float64 | Measured log closure amplitude, dimensionless. |
| `lca_corrupt_sigmas` | `(1, 233)` | float64 | Log closure amplitude uncertainty, dimensionless. |
| `lca_u1` | `(1, 233, 2)` | float64 | First quadrangle baseline $(u,v)$, wavelengths. |
| `lca_u2` | `(1, 233, 2)` | float64 | Second quadrangle baseline $(u,v)$, wavelengths. |
| `lca_u3` | `(1, 233, 2)` | float64 | Third quadrangle baseline $(u,v)$, wavelengths. |
| `lca_u4` | `(1, 233, 2)` | float64 | Fourth quadrangle baseline $(u,v)$, wavelengths. |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Image grid | 64 x 64 | pixels | Square sky brightness field. |
| Pixel pitch | 2.0 | microarcseconds/pixel | Angular separation of adjacent pixel centers. |
| Pixel pitch in radians | 9.696273622188272e-12 | radians/pixel | Acquisition constant used by the Fourier grid; preserve this supplied precision. |
| Pixel coordinates | $l=(31.5-q)\Delta$, $m=(31.5-p)\Delta$ | radians | Row $p$ and column $q$ run from 0 to 63; $\Delta=2$ microarcseconds. |
| Fourier sign | positive | dimensionless | Exponent uses $+2\pi i(ul+vm)$. |
| Pixel response | triangular | dimensionless | Separable triangular pulse in image space. |
| Total source flux | 0.6 | Jy | Known integrated flux for physical output scaling. |
| Observing frequency | 230.0 | GHz | Radio observing frequency. |
| Stations | 7 | stations | Indexed by `station_ids`. |
| Visibilities | 421 | baselines | Samples in this observation. |
| Boundary | finite field | dimensionless | No periodic image extension is assumed by the forward model. |

## Method Hints

Regularized maximum-likelihood imaging can fit closure phases and log closure amplitudes directly while keeping image brightness nonnegative. A Gaussian starting image supplies a smooth flux scale; entropy-like penalties stabilize frequencies the sparse array did not sample. Compare predicted and observed closure quantities with their stated uncertainties. Since closure data alone do not fix absolute position, keep the given image grid and its coordinate convention throughout.

## References

- Chael, A. A., et al. (2018). *Interferometric Imaging Directly with Closure Phases and Closure Amplitudes*. The Astrophysical Journal, 857, 23.
- Thompson, A. R., Moran, J. M., and Swenson, G. W. (2017). *Interferometry and Synthesis in Radio Astronomy*, 3rd ed.
