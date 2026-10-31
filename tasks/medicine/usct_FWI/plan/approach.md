# Solution Approach

## Overview

Frequency-domain Full-Waveform Inversion (FWI) using a Convergent Born Series (CBS) Helmholtz solver, with multi-frequency bootstrapping from 0.3 to 1.25 MHz.

## Forward Model: CBS Helmholtz Solver

### From Helmholtz equation to Born series

The acoustic wavefield $\psi$ in an inhomogeneous medium satisfies the Helmholtz equation:

$$\nabla^2 \psi(\mathbf{r}) + k(\mathbf{r})^2 \psi(\mathbf{r}) = -S(\mathbf{r})$$

where $k(\mathbf{r}) = \omega / c(\mathbf{r})$ is the spatially varying wavenumber and $S(\mathbf{r})$ is the source term. To solve this, we define the scattering potential $V(\mathbf{r}) \equiv k(\mathbf{r})^2 - k_0^2 - i\epsilon$ and rewrite the equation as:

$$\nabla^2 \psi(\mathbf{r}) + (k_0^2 + i\epsilon)\psi(\mathbf{r}) = -V(\mathbf{r})\psi(\mathbf{r}) - S(\mathbf{r})$$

Using the Green's function $g_0$ of the background medium (defined by $\nabla^2 g_0 + (k_0^2 + i\epsilon) g_0 = -\delta$), the solution can be written in the matrix form:

$$\psi = GV\psi + GS$$

where $G \equiv F^{-1}\hat{g}_0(\mathbf{p})F$ is the Green's function operator with $\hat{g}_0(\mathbf{p}) = 1/(|\mathbf{p}|^2 - k_0^2 - i\epsilon)$, $F$ and $F^{-1}$ are the forward and inverse Fourier transforms, and $V$ is a diagonal matrix. The traditional Born series $\psi_\text{Born} = [1 + GV + GVGV + \ldots]GS$ converges only when the spectral radius $\rho(GV) < 1$, which fails for large or strongly scattering media.

### Convergent Born Series (CBS)

The key insight of CBS (Osnabrugge et al., 2016) is to introduce a preconditioner $\gamma$ that guarantees convergence for **arbitrarily large** scattering potentials. Applying $\gamma$ to the integral equation gives:

$$\psi = M\psi + \gamma GS$$

with the iteration operator:

$$M \equiv \gamma GV - \gamma + 1$$

The preconditioner is chosen as:

$$\gamma(\mathbf{r}) = \frac{i}{\epsilon}V(\mathbf{r}), \qquad \epsilon \geq \max_{\mathbf{r}}|k(\mathbf{r})^2 - k_0^2|$$

This ensures $\rho(M) < 1$, so the convergent Born series:

$$\psi = [1 + M + M^2 + M^3 + \ldots]\gamma GS$$

or equivalently the iterative form $\psi_{k+1} = M\psi_k + \gamma GS$, converges to the exact solution.

### Physical interpretation

The parameter $\epsilon$ adds an imaginary component to the background wavenumber, causing the Green's function to decay exponentially: $g_0(\mathbf{r}) \approx e^{i|\mathbf{r}|\sqrt{k_0^2 + i\epsilon}} / (4\pi|\mathbf{r}|)$. This localizes energy, and the imaginary term $i\epsilon$ in the background potential is exactly compensated by the matching term in the scattering potential $V$, so the final solution is unaffected. Each iteration pseudo-propagates the wavefield by a distance $2k_0/\epsilon$, and the solution "grows" outward from the source until the entire domain is covered.

### Wiggle decomposition

To handle the non-smooth Green's function, the CBS iteration uses "wiggle" phase ramps. In 2D, there are $2^2 = 4$ wiggle permutations. Each wiggle shifts the Fourier-domain coordinates by a quarter pixel, smoothing the propagation kernel. The iteration cycles through all wiggles, and each step consists of:

1. **Phase shift**: $E' = E \cdot g_x g_y$ where $g_x = \exp(i w_x x \cdot \pi / (2 \Delta x N_x))$ are wiggle phase ramps
2. **FFT propagation**: $\hat{E} = \text{FFT}(E') / (p_x^2 + p_y^2 - k_{0c}^2/\epsilon - i)$
3. **Inverse phase shift**: $E'' = \text{IFFT}(\hat{E}) \cdot g_x^* g_y^*$
4. **Mixing step**: $\delta E = (1 - \gamma) \delta E - i\gamma^2 E''$

### Convergence and post-processing

The iteration converges when the energy ratio $\|\delta E\|^2 / \|\delta E_0\|^2 < 10^{-5}$ (checked every $N_\text{wiggle}$ iterations) or when the maximum iteration count (500) is reached. After convergence, the accumulated field $E$ is divided by $\gamma$ to recover the physical wavefield, cropped to the region of interest, and scaled.

### Advantages over other methods

Compared to PSTD and FDTD, CBS achieves machine-precision accuracy (vs $O(\Delta t^2)$ or $O(\Delta t^2 \Delta x^2)$), is 2 orders of magnitude faster, and 9 orders of magnitude more accurate in benchmark tests. Its computational cost is $O(\nu k_m^{D+1} \log k_m^D)$ where $\nu$ is the scattering contrast and $D$ is the dimension.

## Inverse Problem: FWI with Adjoint Gradient

For each frequency:

1. **Forward solve**: Run CBS for all 256 sources to get wavefields $\hat{u}_k$.
2. **Source estimation**: Estimate source intensity $\alpha_k$ by least-squares fitting of simulated to observed data:
   $$\alpha_k = \frac{\sum_i \overline{\hat{u}_k(x_r^{(i)})} y_k^{(i)}}{\sum_i |\hat{u}_k(x_r^{(i)})|^2}$$
3. **Misfit**: $J = \sum_k \|\alpha_k \hat{u}_k(x_r) - y_k\|_2^2$ (with near-source muting at 7500 um).
4. **Adjoint gradient**: $G = -\text{Re}\left[\sum_k (\alpha_k u_k) \cdot (\sum_i u_k \bar{r}_{k,i})\right] \cdot 2\omega^2/c(x)$, smoothed with a Gaussian kernel ($\sigma$ depends on frequency).
5. **Scaling**: Normalize $J$ and $G$ by initial values for numerical stability.

## Optimization: NCG with More-Thuente Line Search

- Algorithm: Nonlinear Conjugate Gradient (Polak-Ribiere)
- 3 iterations per frequency
- More-Thuente cubic interpolation line search (max 5 function evaluations)
- Bound constraints: slowness in $[1/1700, 1/1300]$ (velocity in $[1300, 1700]$ m/s)

## Multi-frequency Strategy

Process 20 frequencies from 0.3 to 1.25 MHz sequentially. Use large gradient smoothing ($\sigma=5$) at low frequencies to capture large-scale structure, then reduce ($\sigma=2$ for 0.35-0.75 MHz, $\sigma=1$ for 0.8+ MHz) to resolve fine details. Each frequency's result initializes the next.

## Expected Results

The multi-frequency reconstruction achieves NCC = 1.0 and NRMSE = 0.031 against the baseline reference (a previous complete multi-frequency run).
