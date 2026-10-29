# Seismic Traveltime Tomography: Solution Approach

## Algorithm: Adjoint-State Traveltime Tomography (ATT)

The inversion minimises the least-squares traveltime misfit

```math
\chi(s) = \frac{1}{2} \sum_{n=1}^{N_\text{src}} \sum_{m=1}^{N_\text{rec}}
           \bigl(T^\text{syn}_{nm}(s) - T^\text{obs}_{nm}\bigr)^2
```

by iterative gradient descent on the slowness model s(x, z).

---

## Step 1 — Eikonal Forward Model

For each source n at position (x_n, z_n), solve the 2D isotropic Eikonal equation

```math
\left(\frac{\partial T}{\partial x}\right)^2 + \left(\frac{\partial T}{\partial z}\right)^2 = s^2(x,z)
```

using the **Fast Marching Method** (FMM) via `scikit-fmm`:

```python
phi = -np.ones_like(s)
phi[iz_src, ix_src] = 1          # source node inside zero level set
T = skfmm.travel_time(phi, 1.0/s, dx=[dz, dx])
```

The synthetic traveltime at each receiver is obtained by bilinear interpolation of T.

---

## Step 2 — Sensitivity Kernel by Ray Back-Projection

The Fréchet derivative of χ with respect to slowness is accumulated as a
sensitivity kernel K_s via ray back-projection:

```math
K_s(\mathbf{x}) = \sum_{n,m} R_{nm}
                  \int_{\text{ray}(n \to m)} \delta(\mathbf{x} - \mathbf{y})\, \mathrm{d}\ell
```

where R_{nm} = T^syn_{nm} − T^obs_{nm} is the traveltime residual and the
integral is over the raypath from source n to receiver m.

**Implementation:** rays are traced from each receiver backward to the source
using Euler integration along −∇T / |∇T|.  At each step the kernel and a
coverage-density kernel K_d (which counts total path length regardless of
residuals) are incremented at the nearest grid node.

---

## Step 3 — Kernel Density Normalisation

Raw kernels are dominated by densely sampled regions (near-vertical rays,
crossing zones).  Following Chen et al. (2024), Eq. 18–19:

```math
K_s(\mathbf{x}) \leftarrow \frac{K_s(\mathbf{x})}{\bigl(K_d(\mathbf{x}) + \varepsilon\bigr)^\zeta}
```

with ζ = 0.5 and ε = 10⁻⁴.  ζ = 0 leaves the kernel unchanged; ζ = 1 gives
full density normalisation; ζ = 0.5 balances amplitude correction against
noise amplification in poorly covered zones.

**Gaussian smoothing** (σ = 1.5 grid cells) is applied after normalisation as
implicit regularisation, suppressing high-frequency artefacts from sparse ray
coverage without an explicit TV or Tikhonov term.

---

## Step 4 — Gradient Descent Update

```math
s_{n+1}(\mathbf{x}) = s_n(\mathbf{x}) - \alpha_n \, K_s(\mathbf{x})
```

The step size α_n is chosen so that the maximum fractional slowness change
equals `step_size` = 2%:

```math
\alpha_n = \texttt{step\_size} \times \frac{\max(s_n)}{\max|K_s|}
```

A per-iteration decay `current_step *= step_decay` (step_decay = 0.97) prevents
oscillation as the misfit flattens.  Slowness is clipped to [1/15, 1/1] s/km
after each update to enforce physical velocity bounds (1–15 km/s).

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_iterations` | 40 | Gradient descent iterations |
| `step_size` | 0.02 | Max fractional slowness change per iteration |
| `step_decay` | 0.97 | Per-iteration step size decay factor |
| `zeta` | 0.5 | Kernel density normalisation exponent |
| `epsilon` | 1e-4 | KD normalisation floor |
| `step_km` | 1.0 | Ray tracing Euler step length (km) |
| `smooth_sigma` | 1.5 | Gaussian smoothing sigma (grid cells) |

---

## Marmousi Benchmark

The synthetic dataset uses the Marmousi velocity model (Bourgeois et al., 1991) — a
widely used 2D benchmark representing a sedimentary basin with complex lateral
velocity contrasts.  The model is spatially rescaled to a crustal-scale domain
(184 km × 60 km, 93 × 31 nodes at 2 km spacing) so that the dominant velocity
features (wavelength ~20–60 km) are commensurate with the Fresnel zone of
first-arrival traveltime tomography, making them recoverable by the ray-based
sensitivity kernels.

The true velocity ranges from ~1.5 km/s at the surface to ~5.5 km/s at depth.
The inversion starts from a smooth 1D linear gradient (v0 = 1.5 km/s, v1 = 5.5 km/s)
and uses 120 sources at four depth levels (0, 10, 20, 50 km) with 8 surface receivers.
A correct ATT implementation should achieve NCC ≥ 0.58 on the velocity perturbation
(v_inv − v_background).
