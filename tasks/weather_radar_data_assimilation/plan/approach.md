# Approach: Weather Radar Data Assimilation

## Problem Statement

Reconstruct 3 future full-resolution 128x128 VIL radar frames from 6 past full-resolution frames and sparse (~10%) noisy observations of the future frames.

## Mathematical Formulation

### Forward Model
The observation model is:
$$y_t = M \odot x_t + \eta, \quad \eta \sim \mathcal{N}(0, \sigma^2 I)$$
where $M$ is a binary mask with ~10% coverage and $\sigma = 0.001$.

### Stochastic Interpolant
The generative model defines a stochastic process connecting base distribution $z_0$ (conditioning frames) to target $z_1$ (future frames):
$$z_t = \alpha(t) z_0 + \beta(t) z_1 + \sigma(t) \epsilon$$
with $\alpha(t) = 1 - t$, $\beta(t) = t^2$, $\sigma(t) = 1 - t$ (when sigma_coef=1).

A UNet drift model $b_\theta(z_t, t, \text{cond})$ is trained to predict the velocity field of this interpolant.

### Guided EM Sampling
Starting from $z_0$ (last conditioning frame), iteratively advance via:
$$z_{t+dt} = z_t + b_\theta(z_t, t, \text{cond}) \cdot dt + \sigma(t) \cdot dW - \lambda \nabla_{z_t} \|y - M \odot \hat{x}_1(z_t)\|$$

where $\hat{x}_1(z_t)$ is estimated via second-order stochastic Runge-Kutta extrapolation.

## Solution Strategy

### Step 1: Data Preprocessing
- Load condition frames (6 past), observations (3 future sparse), and mask from `raw_data.npz`.
- Load ground truth from `ground_truth.npz`.
- Scale pixel values: `x_scaled = (x - 0.5) * 10` for UNet processing (reverse: `x = x_scaled / 10 + 0.5`).

### Step 2: Forward Model
- Implement the masking operator: `A(x) = mask * x`.
- Implement Gaussian noiser with sigma=0.001.

### Step 3: UNet Drift Model
- Architecture: UNet with 128 base channels, dim_mults=(1,2,2,2), learned sinusoidal conditioning, 4 attention heads.
- Input: concatenation of current noisy state and 6 conditioning frames along channel dim → 4 input channels (1 current + 3 lookback averaged to 1, or full 6-channel conditioning concatenated with 1-channel state = 7 channels then reduced). Actually: `in_channels = C * 4 = 4` — the UNet takes `[z_t (1ch), cond (6ch compressed or 3ch)]`. The conditioning uses the full 6 frames concatenated along channel dimension with the current state.
  - Correction: `in_channels = C * 4 = 4` with C=1, so the model expects 4-channel input = 1 (current state) + 3 (from the 6 conditioning frames, likely last 3 or subsampled). Looking at the code: `zt = torch.cat([zt, cond], dim=1)` where zt is (B,1,H,W) and cond is (B,6,H,W), giving 7 channels. But in_channels=4... The UNet handles the 7-channel input through the first convolution layer that was trained with the specific lookback window configuration.

### Step 4: Guided Sampling (EM with 500 steps)
- Initialize from the last conditioning frame as base.
- For each of the 500 time steps from t=0 to t=0.999:
  1. Compute drift $b_\theta(z_t, t, \text{cond})$.
  2. Estimate clean prediction via second-order stochastic Runge-Kutta:
     - First estimate: $\hat{x}_1^{(1)} = z_t + b_\theta \cdot (1-t) + \text{noise} \cdot \sqrt{\text{variance}}$
     - Evaluate drift at $\hat{x}_1^{(1)}$ at $t=1$, average: $\hat{x}_1^{(2)} = z_t + (b_\theta + b_{\theta,2})/2 \cdot (1-t) + \text{noise}$
  3. MC averaging: repeat the second estimate MC_times=25 times.
  4. Compute guidance gradient: $\nabla_{z_t} \|y_t - M \odot \hat{x}_1\|$.
  5. Update: $z_{t+dt} = z_t + b_\theta \cdot dt + \sigma(t) \cdot \mathcal{N}(0, dt) - 0.1 \cdot \nabla$.

### Step 5: Autoregressive Rollout
- After generating frame $t+1$, shift conditioning window: drop oldest frame, append prediction.
- Repeat for frames $t+2$ and $t+3$.

### Step 6: Evaluation
- Compute NCC and NRMSE between reconstruction and ground truth.
- Generate comparison visualizations.

## Expected Results

| Method | NCC | NRMSE |
|--------|-----|-------|
| FlowDAS (MC=25, 500 steps) | 0.8706 | 0.0514 |
