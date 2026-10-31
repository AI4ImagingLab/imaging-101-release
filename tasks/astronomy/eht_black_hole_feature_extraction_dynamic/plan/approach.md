# Approach: Dynamic Feature Extraction via Per-Frame α-DPI

## Problem Statement

Given a time-varying EHT observation of a black hole (simulating Sgr A*-like variability), recover the posterior distribution of geometric crescent parameters at each time snapshot independently.

## Algorithm: Per-Frame α-DPI

Each frame is processed independently using the α-DPI algorithm (Sun et al. 2022):

### Step 1: Training

For each frame $t$, minimize the α-divergence between the variational posterior $q_\theta(\mathbf{x})$ (parameterized by a Real-NVP normalizing flow) and the true posterior $p(\mathbf{x}|\mathbf{y}_t)$:

$$\mathcal{L}(\theta) = D_\alpha(p \| q_\theta) \approx \sum_i w_i^\alpha \cdot \ell_i$$

where $\ell_i$ is the data fidelity loss (closure phase + log closure amplitude χ²) for sample $\mathbf{x}_i \sim q_\theta$.

With α = 1 and β = 0 (our default), this reduces to KL divergence:

$$\mathcal{L}(\theta) = \mathbb{E}_{q_\theta}[\ell(\mathbf{x})] - \log|\det J_\theta|$$

### Step 2: Importance Sampling

After training, draw N = 10000 samples from $q_\theta$ and reweight by:

$$w_i \propto \frac{p(\mathbf{y}_t|\mathbf{x}_i) \, p(\mathbf{x}_i)}{q_\theta(\mathbf{x}_i)}$$

### Step 3: Parameter Extraction

Convert flow output (sigmoid-transformed to [0,1]) to physical units via linear rescaling:
- Diameter: 2 × (r_range[0] + p[0] × (r_range[1] - r_range[0]))
- Width: width_range[0] + p[1] × (width_range[1] - width_range[0])
- Asymmetry: p[2] (already in [0,1])
- PA: 362 × p[3] - 181 (degrees)

## Data-Dependent Warmup

The data fidelity weight follows a log-linear schedule:

$$w(k) = \min(10^{-\text{start\_order} + k / \text{decay\_rate}}, 1)$$

This prevents the flow from collapsing early when the closure quantity loss dominates.

## Key Design Choices

1. **Independent frames** — no temporal coupling between frames. This is deliberate: we test whether α-DPI can recover time-varying parameters from snapshot data alone.

2. **SimpleCrescent (4 params)** — minimal model for efficiency. Each frame trains in ~2-5 min on GPU, making 10 frames feasible in ~30-60 min total.

3. **EHT 2017 array** — 8 stations with realistic UV coverage that varies across the observation window due to Earth rotation.

4. **Closure quantities only** — gain-invariant, avoiding station-based calibration uncertainties.
