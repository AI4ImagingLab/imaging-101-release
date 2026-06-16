# Weather Radar Data Assimilation

> Reconstruct full-resolution future weather radar frames from sparse point observations and past context, combining a learned generative prior with physics-based data assimilation.

> Domain: earth science | Keywords: data assimilation, generative modeling, radar nowcasting | Difficulty: Hard

## Background

Weather nowcasting aims to predict near-future precipitation patterns from radar observations. The Storm Event Imagery (SEVIR) dataset provides Vertically Integrated Liquid (VIL) radar mosaics over the contiguous United States, capturing precipitation intensity at regular spatial and temporal intervals. In operational settings, observations are often sparse due to sensor failures, limited coverage, or bandwidth constraints, making direct prediction from incomplete data an ill-posed inverse problem.

Data assimilation fuses incomplete observations with a dynamical prior to produce the best estimate of the atmospheric state. Classical methods (e.g., variational or ensemble Kalman filters) rely on hand-crafted dynamical models. Modern approaches learn the dynamics from data, using score-based or flow-based generative models as priors. The generative model captures the joint distribution of weather states, and guided sampling incorporates sparse observations at inference time.

## Problem Description

The forward measurement model is a sparse masking operator with additive noise:

$$\mathbf{y}_t = \mathbf{M} \odot \mathbf{x}_t + \boldsymbol{\eta}, \quad \boldsymbol{\eta} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$

where:
- $\mathbf{x}_t \in \mathbb{R}^{128 \times 128}$ is the full-resolution VIL radar frame at time $t$
- $\mathbf{M} \in \{0, 1\}^{128 \times 128}$ is a binary observation mask with approximately 10% coverage
- $\sigma = 0.001$ is the observation noise standard deviation
- $\mathbf{y}_t$ is the observed (masked + noisy) frame

Given 6 past full-resolution frames $\{\mathbf{x}_{t-5}, \ldots, \mathbf{x}_t\}$ (condition) and sparse observations of the next 3 frames $\{\mathbf{y}_{t+1}, \mathbf{y}_{t+2}, \mathbf{y}_{t+3}\}$, the goal is to reconstruct the full-resolution future frames $\{\hat{\mathbf{x}}_{t+1}, \hat{\mathbf{x}}_{t+2}, \hat{\mathbf{x}}_{t+3}\}$.

The problem is ill-posed because only ~10% of pixels are observed at each future time step, and weather dynamics are chaotic and nonlinear. The autoregressive structure (each predicted frame conditions the next) propagates errors, making accurate multi-step reconstruction challenging.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `condition_frames` | `(1, 6, 128, 128)` | float32 | Past VIL radar frames used as conditioning context, pixel values in [0, 1] |
| `observations` | `(1, 3, 128, 128)` | float32 | Sparse noisy observations of 3 future frames, masked + Gaussian noise |
| `observation_mask` | `(1, 1, 128, 128)` | float32 | Binary observation mask (same spatial mask for all 3 frames), ~10% coverage |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `target_frames` | `(1, 3, 128, 128)` | float32 | Full-resolution ground truth VIL radar frames, pixel values in [0, 1] |

### data/meta_data.json

Contains imaging parameters: image dimensions (128x128), number of condition/prediction frames (6/3), mask ratio (0.1), noise sigma (0.001), and SEVIR-LR dataset metadata (4 km spatial resolution, 10 min temporal resolution).

## Method Hints

**Flow-based data assimilation** uses a stochastic interpolant framework to learn the dynamics of weather radar sequences. A UNet drift model is trained on the joint distribution of past-and-future frame pairs using a flow-matching objective. At inference time, guided Euler-Maruyama (EM) sampling incorporates the sparse observations through gradient-based guidance:

1. Sample initial noise and iteratively denoise using the learned drift model, conditioned on past frames.
2. At each EM step, estimate the clean prediction via Taylor expansion (second-order stochastic Runge-Kutta), compute the data-fidelity gradient between the estimate and observations, and apply a guidance correction.
3. Autoregressive rollout: after generating frame $t+1$, shift the conditioning window to include it and predict $t+2$, then $t+3$.

Key algorithmic components:
- **Stochastic interpolant**: defines a path between noise and data via $z_t = \alpha(t) z_0 + \beta(t) z_1 + \sigma(t) \epsilon$ with $\beta(t) = t^2$
- **Guided sampling**: gradient of the observation likelihood $\|\mathbf{y} - \mathbf{M} \odot \hat{\mathbf{x}}_1\|$ steers the sampling toward consistency with measurements
- **Monte Carlo averaging**: multiple forward estimates are averaged to reduce variance in the guidance gradient

## References

1. Chen, S., et al. "FlowDAS: Flow-based Data Assimilation." 2025.
2. SEVIR Dataset: https://proceedings.neurips.cc/paper/2020/hash/fa78a16157fed00d7a80515818f2ef00-Abstract.html
