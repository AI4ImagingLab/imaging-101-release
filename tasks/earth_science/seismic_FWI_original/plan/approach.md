# Approach: 2D Acoustic FWI with Scratch C-PML Wave Propagation

## Problem

Given observed seismograms $d^{\text{obs}}$ (pressure recorded at receivers for each shot), invert for the subsurface P-wave velocity model $v(\mathbf{x})$ that minimizes:

$$J(v) = \frac{1}{2}\sum_{s}\left\|\mathcal{T}[F_s(v)] - \mathcal{T}[d^{\text{obs}}_s]\right\|_2^2$$

where $F_s(v)$ is the forward-modeled data for shot $s$ and $\mathcal{T}$ is a cosine taper applied to the last 5 samples of each trace.

---

## Dataset: Marmousi Model

- Full model: 2301×751 cells at 4 m spacing (≈ 9.2 km × 3.0 km).
- Preprocessed to 461×151 at 20 m (5× Gaussian smooth + subsample).
- 10 shots uniformly distributed across surface; 93 receivers spanning the model.
- 1350 time samples at $\Delta t = 4$ ms (total 5.4 s).
- Ricker source wavelet: 5 Hz central frequency, peak at $t_p = 0.3$ s.

---

## Physics: 2D Acoustic Wave Equation

$$\frac{\partial^2 p}{\partial t^2} = v^2(\mathbf{x})\,\nabla^2 p - v^2(\mathbf{x})\,s(t)\,\delta(\mathbf{x}-\mathbf{x}_s)$$

Time integration: 2nd-order Verlet scheme (leap-frog):

$$p^{n+1} = v^2\,\Delta t_{\text{inner}}^2\,W(p^n, \boldsymbol\psi^n, \boldsymbol\zeta^n) + 2p^n - p^{n-1}$$

Space derivatives: 4th-order central finite differences.

---

## Absorbing Boundaries: Convolutional PML (C-PML)

Reference: Pasalic & McGarry (2010), SEG.

### Profile construction

For each spatial dimension $d$ with PML width $L$ (meters):

$$\sigma_0 = -\frac{(1+p)\,v_{\max}\,\ln R}{2L}, \quad \alpha_0 = \pi f_{\text{pml}}$$

At each grid point, the normalized PML fraction $\eta \in [0,1]$ (distance into PML / PML width):

$$\sigma = \sigma_0\,\eta^p, \quad \alpha = \alpha_0\,(1-\eta)$$

$$a = \exp\!\left(-(\sigma+\alpha)\,|\Delta t_{\text{inner}}|\right), \quad b = \frac{\sigma}{\sigma+\alpha}(a-1)$$

Interior points: $a=0$, $b=0$ (PML inactive). Parameters: $p=2$, $R=10^{-3}$, 20 cells PML on each side.

### C-PML wave step

Auxiliary variables $\psi_d$, $\zeta_d$ per dimension:

$$\text{tmp}_d = (1+b_d)\frac{\partial^2 p}{\partial d^2} + \frac{\partial b_d}{\partial d}\frac{\partial p}{\partial d} + \frac{\partial}{\partial d}(a_d\psi_d)$$

$$W = \sum_d \left[(1+b_d)\,\text{tmp}_d + a_d\,\zeta_d\right]$$

$$\psi_d^{n+1} = b_d\frac{\partial p^n}{\partial d} + a_d\psi_d^n, \quad \zeta_d^{n+1} = b_d\,\text{tmp}_d + a_d\zeta_d^n$$

In the interior ($a_d=b_d=0$): $W$ reduces to the standard Laplacian.

---

## CFL Condition

To ensure numerical stability:

$$\Delta t_{\text{inner}} \leq \frac{0.6}{\sqrt{1/\Delta y^2 + 1/\Delta x^2}\cdot v_{\max}}$$

The user time step $\Delta t$ is subdivided: $\text{step\_ratio} = \lceil\Delta t / \Delta t_{\max}\rceil$, $\Delta t_{\text{inner}} = \Delta t / \text{step\_ratio}$.

Source amplitudes are FFT-upsampled from $N_t$ to $N_t \cdot \text{step\_ratio}$ samples. Receiver data is FFT-downsampled back to $N_t$ after simulation.

---

## Source Injection

Consistent with deepwave's sign convention:

$$p^{n+1}[\mathbf{x}_s] \mathrel{+}= -v(\mathbf{x}_s)^2\,\Delta t_{\text{inner}}^2\,s(t_n)$$

where $s(t)$ is the upsampled Ricker wavelet. The $v(\mathbf{x}_s)^2$ factor is computed differentiably so the FWI gradient includes source-velocity coupling.

---

## Ricker Wavelet

$$s(t) = \left(1 - 2\pi^2 f^2(t-t_p)^2\right)\exp\!\left(-\pi^2 f^2(t-t_p)^2\right)$$

with $f=5$ Hz, $t_p = 1.5/f = 0.3$ s.

---

## FWI Optimization

- **Optimizer**: Adam, $\text{lr}=100$.
- **Learning rate schedule**: MultiStepLR, milestones=[75, 300], $\gamma=0.5$.
- **Gradient post-processing** (every iteration):
  1. Gaussian smooth the gradient: $\sigma = 1$ grid point.
  2. Clip at 98th percentile of $|\nabla J|$.
- **Physical constraint**: $v \in [1480, 5800]$ m/s (clamped after each update).
- **Epochs**: 800.

---

## Autograd and Memory

Gradients are computed automatically via PyTorch autograd. Each shot builds an independent computation graph (processed sequentially). Peak GPU memory per shot is approximately:

$$5 \times N_{y,p} \times N_{x,p} \times N_{t,\text{inner}} \times 4\text{ bytes}$$

For $N_{y,p}=505$, $N_{x,p}=195$, $N_{t,\text{inner}}\approx 2700$ (step_ratio=2): **~5.3 GB** per shot.

---

## Evaluation Metrics

- **Velocity relative L2 error**: $\|v_{\text{inv}} - v_{\text{true}}\|_2 / \|v_{\text{true}}\|_2$. Target: ≤ 10%.
- **Data relative L2 error per shot**: $\|F_s(v_{\text{inv}}) - d^{\text{obs}}_s\|_2 / \|d^{\text{obs}}_s\|_2$. Target: ≤ 2%.
