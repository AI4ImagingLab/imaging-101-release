---
name: oracle-cars-spectroscopy
description: "Recover N2 gas temperature from a synthetic coherent anti-Stokes Raman scattering (CARS) intensity spectrum by nonlinear least-squares fitting of a G-matrix forward model. Provides the data contract, forward physics (Dunham line positions, modified-exponential-gap relaxation matrix, eigendecomposition susceptibility, non-resonant background, pump/slit convolutions, downsampling), the trust-region solver setup, and the output writing/validation conventions needed to reproduce the reference behavior."
---

# oracle-cars-spectroscopy Oracle Skill

This skill reconstructs gas temperature from a measured normalized CARS spectrum I(nu) on a wavenumber axis. The measured spectrum is loaded and normalized, then a physics forward model I_model(T, x_mol, nu_shift, slit_width) is fit to it with scipy trust-region-reflective least squares; the estimated temperature and the best-fit reconstructed spectrum are written to a per-case .npz file.

Each numbered section below is an independent, self-contained unit: it states its own constants, formulas, and code-level conventions so it can be applied or omitted on its own. Use the sections in order — contract and data conventions first, then the forward-model physics, then the solver, then output writing and checks. Constants (spectroscopic Dunham terms, the four-parameter relaxation fit, universal constants, non-resonant susceptibility table) all come from the per-case params JSON so the model can be re-parameterized without editing code.

Runtime: use the run-local Python interpreter (for example `public/envs/runtime/.venv-posix/bin/python`) available on PATH. Read inputs only from `public/`, write scratch to `workspace/`, and write submissions to `outputs/`. Bundled helper files live under `${CLAUDE_SKILL_DIR}`.

## Spectroscopic, universal, and relaxation constant tables

## Spectroscopic, universal, and relaxation constant tables

The forward model consumes three constant blocks from the params JSON.

`molecular_constants.N2` (used by line-position, population, and cross-section formulas):
- Vibrational: `we=2358.518`, `wx=14.2935`, `wy=-0.00592949`, `wz=-0.00024`.
- Rotational: `Be=1.99826`, `De=5.774e-06`, `alpha_e=0.0173035`, `beta_e=1.55e-08`, `gamma_e=-3.1536099e-05`, `H0=3e-12`, `He=1.8e-12`.
- `v0=2330.7`, `mu=0.31`, `MW=28.013`, `Const_Raman=0.24158`, `G/A=1.166` (polarizability ratio), `Const_Osc=195853800.0`.
- `fit_param_relaxation_matrix = [alpha, beta, sigma, m] = [0.0231, 1.67, 1.21, 0.1487]` (modified-exponential-gap law).

`universal_constants`: `h=6.62607e-34`, `h_bar=1.05457e-34`, `c=29979000000.0` (cm/s), `k=1.380649e-16`, `k_=0.695035`, `hc/k=1.44`, `R=8.314`, `Const_D=7.162e-7`, `Const_N=0.724`. The value `1.44` (= hc/k in cm·K) appears throughout the Boltzmann exponents.

`chi_nrs["SET 1"]`: `T0=273.15`, `P0=1.01325`, `SPECIES` non-resonant values including `N2: 7.7877` (others CO2, CO, H2, O2). Used for the non-resonant background scaling.

Access pattern in code: `MOL_CONST = params['molecular_constants']`, `UNIV_CONST = params['universal_constants']`, `CHI_NRS = params['chi_nrs']`, then `mc = MOL_CONST['N2']`.

## G-matrix eigendecomposition resonant susceptibility

## G-matrix eigendecomposition resonant susceptibility

Compute the complex resonant susceptibility `chi_rs(nu)` on a fine wavenumber grid.

Fine grid: apply optional axis calibration `nu_expt_mod = nu_expt * nu_stretch + nu_shift`, then
```
step = 0.05
nu_s = arange(nu_expt_mod[0]-10, nu_expt_mod[-1]+10, step)
```

Loop over branches `(0, +2, -2)` and vibrational levels `v in range(2)` with `js=30`, `_js=arange(js)`:
```
nu_raman = line_pos(v, _js, branch)                 # (30,)
K = diag(nu_raman) + gamma_mat*1j                    # G-matrix
eigvals, eigvec = np.linalg.eig(K)
eigvec_inv = np.linalg.inv(eigvec)
del_pop = pop_factor(T, v, _js, branch)              # (30,)
pt_coeff, cd_coeff via int_corr for each j
pol_ratio = mc['G/A'] ; Const_Raman = mc['Const_Raman']
if branch in (2,-2): d2 = Const_Raman**2 * (4/45)*pt_coeff*pol_ratio**2 * (v+1) * cd_coeff
else:                 d2 = Const_Raman**2 * (1 + (4/45)*pt_coeff*pol_ratio**2) * (v+1) * cd_coeff
d = d2**0.5
term_l = d @ eigvec                                  # (30,)
term_r = eigvec_inv @ diag(del_pop) @ d              # (30,)
term   = term_l * term_r
for jj in range(js):
    denom = (-nu_s + real(eigvals[jj]))**2 + imag(eigvals[jj])**2
    chi_rs += 0.5 * term[jj] * conj(-nu_s + eigvals[jj]) / denom
```
After both loops: `chi_rs = chi_rs / (2*pi*c)` with `c` = universal `c` (2.9979e10 cm/s).

This is the Kronig-Kramers / G-matrix mixed-line sum: overlapping rotational lines are mixed through the eigenvectors, producing collisional narrowing. Preserve the `0.5` prefactor, the `(-nu_s + eigval)` sign convention, and the branch/vibrational loop ranges.

## Pump-linewidth and slit convolutions plus grid downsampling

## Pump-linewidth and slit convolutions plus grid downsampling

Apply instrumental broadening to `I_as` on the fine grid, then resample to the experimental axis.

Pump laser linewidth (Kataoka approximation) — only if `pump_lw > 0`:
```
gaussian_line(w,w0,sigma) = 2/sigma*(ln2/pi)**0.5 * exp(-4*ln2*((w-w0)/sigma)**2)   # 0 if sigma==0
pump_ls = gaussian_line(nu_s, (nu_s[0]+nu_s[-1])/2, pump_lw)
chi_convol = np.convolve(chi_total, pump_ls, 'same')
d_nu = nu_s[1]-nu_s[0]
I_as = 0.5*(I_as + d_nu*np.abs(chi_convol)**2)
```

Instrument slit function = asymmetric Gaussian centered on grid midpoint, params `[sigma, k, a_sigma, a_k]` (defaults `[0.5, 2.0, 0, 0]`, i.e. slit_width and slit_shape_factor):
```
asym_Gaussian(w,w0,sigma,k,a_sigma,a_k,offset=0):
  low  = exp(-abs((w[w<=w0]-w0)/(sigma-a_sigma))**(k-a_k))
  high = exp(-abs((w[w>w0]-w0)/(sigma+a_sigma))**(k+a_k))
  resp = (append(low,high)+offset)**power_factor   # power_factor=1
  return nan_to_num(resp/resp.max())
slit = asym_Gaussian(nu_s, (nu_s[0]+nu_s[-1])/2, slit_width, 2.0, 0, 0)
I_as = np.convolve(I_as, slit, 'same')
```

Downsample fine grid to experimental grid via local-mean:
```
downsample(w, w_fine, spec_fine, mode='local_mean'):
  hw = max(1, int((w[1]-w[0])/(w_fine[1]-w_fine[0])/2))
  idx = clip(searchsorted(w_fine, w), 0, len(w_fine)-1)
  out[i] = mean(spec_fine[max(0,idx-hw): idx+hw+1])   # else spec_fine[idx]
I_final = downsample(nu_expt_mod, nu_s, I_as, 'local_mean')
if I_final.max()>0: I_final /= I_final.max()   # final peak normalization -> (200,)
```

The returned `(200,)` peak-normalized `I_final` is the model spectrum compared against the measured signal. Keep the `0.5*(...)` averaging of unbroadened and pump-broadened intensity, the `'same'` convolution mode, and the final peak normalization.

## Trust-region least-squares temperature inversion

## Trust-region least-squares temperature inversion

Fit the forward model to the preprocessed measured signal with `scipy.optimize.least_squares` (TRF).

Free parameter order and initial guess `x0 = [temperature, x_mol, nu_shift, slit_width]`:
```
x0 = [2000.0, 0.79, 0.0, 0.5]
bounds_lower = [300.0, 0.1, -5.0, 0.01]
bounds_upper = [3500.0, 1.0,  5.0, 5.0]
```
Note the initial temperature 2000 K is intentionally offset from the true value to test convergence.

Residual function builds the forward-model param dict per evaluation:
```
def residual(x):
    T, x_mol, nu_shift, slit_width = x
    params = {'nu': nu_axis, 'temperature': T, 'pressure': 10.0, 'x_mol': x_mol,
              'species': 'N2', 'pump_lw': 1.0, 'nu_shift': nu_shift, 'nu_stretch': 1.0,
              'slit_params': [slit_width, 2.0, 0, 0]}
    return forward_operator(params) - measured_signal
result = least_squares(residual, x0, bounds=(bounds_lower, bounds_upper), method='trf', max_nfev=100)
```
Fixed (not fitted): `pressure=10.0` bar, `pump_lw=1.0`, `nu_stretch=1.0`, slit shape factor `2.0`. `max_nfev=100`.

After fitting, extract `best = result.x`, set `best_params = {temperature, x_mol, nu_shift, slit_width}`, and regenerate the best-fit spectrum by one more `forward_operator` call with those params. That spectrum is `reconstructed_spectrum`; `best[0]` is `estimated_temperature_K`.

## Write per-case submission NPZ

## Write per-case submission NPZ

Write `outputs/<case_id>.npz` with the three required arrays, matching shapes/dtype exactly and ensuring finiteness.

```python
import numpy as np, os
os.makedirs('outputs', exist_ok=True)
T_pred = float(best_params['temperature'])
y_pred = np.asarray(reconstructed_spectrum, dtype=np.float64).reshape(200)
nu_out = np.asarray(nu_axis, dtype=np.float64).reshape(200)
assert np.all(np.isfinite(y_pred)) and np.isfinite(T_pred)
np.savez('outputs/%s.npz' % case_id,
         estimated_temperature_K=np.array([T_pred], dtype=np.float64),  # shape (1,)
         reconstructed_spectrum=y_pred,                                 # shape (200,)
         nu_axis=nu_out)                                                # shape (200,)
```

Keys must be exactly `estimated_temperature_K` (shape `(1,)`), `reconstructed_spectrum` (shape `(200,)`), `nu_axis` (shape `(200,)`). Use float32/float64 only; the grader does not cast. `nu_axis` must equal the input axis. A reusable writer helper is bundled at `${CLAUDE_SKILL_DIR}/scripts/script_001.py`.
