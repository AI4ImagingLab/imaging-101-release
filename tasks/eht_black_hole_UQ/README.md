# EHT Black Hole Probabilistic Imaging (DPI)

> Learn the posterior distribution over radio images of a supermassive black hole from sparse interferometric measurements, enabling principled uncertainty quantification via a normalizing flow generative model.

> Domain: Astronomy | Keywords: radio interferometry, variational inference | Difficulty: Hard

---

## Background

Standard image reconstruction methods (CLEAN, RML) produce a single point
estimate. But the EHT inverse problem is severely ill-posed: many images are
consistent with the sparse measurements. **Which image features are reliable,
and which are artifacts of the reconstruction?**

Deep Probabilistic Imaging (DPI) addresses this by training a Real-NVP
normalizing flow to approximate the posterior distribution p(image | data).
After training, posterior samples are obtained instantly by pushing random
Gaussian vectors through the learned flow, enabling:

- **Uncertainty maps** (pixel-wise standard deviation)
- **Multi-modal exploration** (distinct plausible images)
- **Feature credibility** (is the asymmetry real?)

---

## Problem Description

The forward model maps images to interferometric observables:

$$V_{ij} = \sum_p I_p \, e^{-2\pi i (u_{ij} l_p + v_{ij} m_p)}$$

Because station-based gain errors corrupt raw visibilities, DPI uses
**closure quantities** that are immune to gain calibration:

- **Closure phase** (triangle of 3 stations):
  $$\phi_{ijk} = \arg(V_{ij}) + \arg(V_{jk}) + \arg(V_{ki})$$

- **Log closure amplitude** (quadrangle of 4 stations):
  $$\ln A_{ijkl} = \ln|V_{ij}| + \ln|V_{kl}| - \ln|V_{ik}| - \ln|V_{jl}|$$

The inverse problem is to recover the full posterior distribution $p(\mathbf{x} \mid y)$ over images consistent with these closure measurements — a multi-modal distribution that point-estimate methods cannot characterize.

**Input:** 465 closure phases and 485 log closure amplitudes from a real SgrA* observation.
**Output:** a generative model that produces posterior image samples.

---

## Data Description

### `data/raw_data.npz`

Pre-processed observation arrays from a real SgrA* 2015 ALMA VLBI observation:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `vis` | (938,) | complex128 | Complex visibilities |
| `sigma` | (938,) | float64 | Per-visibility thermal noise sigma (Jy) |
| `uv_coords` | (938, 2) | float64 | Baseline (u, v) coordinates in wavelengths |
| `times` | (938,) | float64 | Observation times |
| `t1`, `t2` | (938,) | bytes | Station names for each baseline |
| `station_ids` | (938, 2) | int64 | Integer station pair indices |
| `cp_times` | (465,) | float64 | Closure phase timestamps |
| `cp_t1`, `cp_t2`, `cp_t3` | (465,) | bytes | Closure triangle station names |
| `cp_values_deg` | (465,) | float64 | Observed closure phases (degrees) |
| `cp_sigmas_deg` | (465,) | float64 | Closure phase sigmas (degrees) |
| `lca_times` | (485,) | float64 | Log closure amplitude timestamps |
| `lca_t1`, `lca_t2`, `lca_t3`, `lca_t4` | (485,) | bytes | Closure quad station names |
| `lca_values` | (485,) | float64 | Observed log closure amplitudes |
| `lca_sigmas` | (485,) | float64 | Log closure amplitude sigmas |

> **Note**: `raw_data.npz` was created from the original `obs.uvfits` file using
> `ehtim >= 1.2` and `astropy >= 4.0`. Those packages are NOT required at runtime;
> they are only needed if you want to re-generate `raw_data.npz` from the original
> FITS/UVFITS files (see `src/generate_data.py`).

### `data/ground_truth.npz`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `image` | (32, 32) | float64 | Ground-truth image regridded to 32×32, 160 μas FOV |

> Regridded from the original 128×128 FITS reference image using ehtim.

### `data/meta_data.json`

JSON file with imaging and training parameters:
- `npix` = 32: image size (32×32 pixels)
- `fov_uas` = 160.0: field of view in microarcseconds
- `prior_fwhm_uas` = 50.0: Gaussian prior FWHM in microarcseconds
- `n_flow` = 16: number of Real-NVP flow blocks
- `n_epoch` = 10000: training epochs
- `batch_size` = 32: mini-batch size
- `lr` = 1e-4: Adam learning rate
- `logdet_weight` = 1.0: entropy term weight
- `l1_weight`, `tsv_weight`, `flux_weight`, `center_weight`, `mem_weight`: prior weights
- `grad_clip` = 0.1: gradient clipping threshold

---

## Method Hints

**Normalizing flows for posterior estimation.** A normalizing flow $G_\theta: z \mapsto x$ transforms a simple Gaussian prior on latent code $z$ into a complex image distribution. Training minimizes the KL divergence from the true posterior:

$$\theta^* = \arg\min_\theta \, \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ \mathcal{L}_\text{data}(y, G_\theta(z)) + \lambda \mathcal{R}(G_\theta(z)) - \beta \log\left|\det \frac{\partial G_\theta}{\partial z}\right| \right]$$

where $\mathcal{L}_\text{data}$ is a chi-squared loss on closure quantities, $\mathcal{R}(x)$ combines image regularizers (maximum entropy, total squared variation, flux and centroid constraints), and the log-determinant term is an entropy bonus that prevents posterior collapse.

**Why flows suit this problem.** After training, posterior samples are obtained in a single forward pass through $G_\theta$, making uncertainty quantification orders of magnitude faster than MCMC. The architecture should be a Real-NVP-style coupling flow mapping $\mathbb{R}^{N^2} \to \mathbb{R}^{N^2}$, with a positivity-enforcing output activation.

**Closure-only data fidelity.** Using only closure phases and log closure amplitudes (rather than raw visibilities) makes training robust to station-based gain errors that dominate EHT calibration uncertainty.

---

## References

- Sun, H. & Bouman, K.L. (2020). *Deep Probabilistic Imaging: Uncertainty Quantification and Multi-modal Solution Characterization for Computational Imaging*. arXiv:2010.14462.
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). *Density estimation using Real-NVP*. ICLR 2017.
- EHT Collaboration et al. (2019). *First M87 Event Horizon Telescope Results*. ApJL, 875.
- Chael, A.A. et al. (2018). *Interferometric Imaging Directly with Closure Phases and Closure Amplitudes*. ApJ, 857, 23.
