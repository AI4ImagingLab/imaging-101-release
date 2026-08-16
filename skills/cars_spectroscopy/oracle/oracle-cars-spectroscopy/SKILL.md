---
name: oracle-cars-spectroscopy
description: "Recover N2 gas temperature from a synthetic coherent anti-Stokes Raman scattering (CARS) intensity spectrum by nonlinear least-squares fitting of a G-matrix forward model. Provides the data contract, forward physics (Dunham line positions, modified-exponential-gap relaxation matrix, eigendecomposition susceptibility, non-resonant background, pump/slit convolutions, downsampling), the trust-region solver setup, and the output writing/validation conventions needed to reproduce the reference behavior."
---

# oracle-cars-spectroscopy Oracle Skill

This skill reconstructs gas temperature from a measured normalized CARS spectrum I(nu) on a wavenumber axis. The measured spectrum is loaded and normalized, then a physics forward model I_model(T, x_mol, nu_shift, slit_width) is fit to it with scipy trust-region-reflective least squares; the estimated temperature and the best-fit reconstructed spectrum are written to a per-case .npz file.

Each numbered section below is an independent, self-contained unit: it states its own constants, formulas, and code-level conventions so it can be applied or omitted on its own. Use the sections in order — contract and data conventions first, then the forward-model physics, then the solver, then output writing and checks. Constants (spectroscopic Dunham terms, the four-parameter relaxation fit, universal constants, non-resonant susceptibility table) all come from the per-case params JSON so the model can be re-parameterized without editing code.

Runtime: use the run-local Python interpreter (for example `public/envs/runtime/.venv-posix/bin/python`) available on PATH. Read inputs only from `public/`, write scratch to `workspace/`, and write submissions to `outputs/`. Bundled helper files live under `${CLAUDE_SKILL_DIR}`.

<!-- ORACLE_OP_START op_010_public_contract -->
## Public task contract and submission schema

## Public task contract and submission schema

Goal: given a measured normalized CARS intensity spectrum and its wavenumber axis for a case, estimate the gas temperature (K) and produce the best-fit reconstructed spectrum.

Case discovery: read the case list from `public/cases.json` (key `cases`, each entry has `id`, `input_dir`, `expected_output`, `files`, `params`). For each case id (e.g. `case_000`) the inputs live under `public/cases/<case_id>/input_data/`.

Input per case:
- `raw_data.npz` with `measurements` shape `(1, 200)` float64 (noisy normalized intensity, dimensionless) and `nu_axis` shape `(1, 200)` float64 (wavenumber in cm^-1).
- `params_data.json` with physical parameters and constants.

Output per case: write `outputs/<case_id>.npz` (base = run root) containing exactly:
- `estimated_temperature_K`: shape `(1,)`, float32 or float64.
- `reconstructed_spectrum`: shape `(200,)`, float32/float64.
- `nu_axis`: shape `(200,)`, float32/float64, aligned with `reconstructed_spectrum`.

Hard invariants: all written arrays must be finite (no NaN/Inf); no dtype casting is performed by the grader (`allow_cast=false`) so match shapes and use a float dtype. Keep `nu_axis` identical to the input axis (1-D, length 200). One visible case (`case_000`) is provided; the same protocol applies to any hidden cases.
<!-- ORACLE_OP_END op_010_public_contract -->

<!-- ORACLE_OP_START op_020_data_loading -->
## Load raw arrays and per-case parameters

## Load raw arrays and per-case parameters

Load the case NPZ and JSON with the run-local Python:

```python
import numpy as np, json
raw = np.load('public/cases/<case_id>/input_data/raw_data.npz')
measurements = raw['measurements'][0]   # (200,)  batch-first -> take row 0
nu_axis      = raw['nu_axis'][0]        # (200,)
params = json.load(open('public/cases/<case_id>/params_data.json'))
```

Batch-first convention: stored arrays have a leading singleton axis of size 1; index `[0]` to get the 1-D `(200,)` vectors. Keep `nu_axis` unchanged for output.

From `params` read the scalar physical settings and constant tables:
- `species` (e.g. `"N2"`), `wavenumber_range_cm-1` `[2280, 2330]`, `num_spectral_points` `200`.
- `pressure_bar` (10.0), `pump_linewidth_cm-1` (1.0), `slit_width_cm-1` (0.5), `slit_shape_factor` (2.0), `noise_level` (0.02), `random_seed` (42).
- `molecular_constants.<species>` (Dunham/spectroscopic constants + relaxation fit), `universal_constants`, `chi_nrs`.

These three constant tables feed the forward model. Do not hardcode them; pull them from the JSON so a different species/pressure can be handled by reading different values.
<!-- ORACLE_OP_END op_020_data_loading -->

<!-- ORACLE_OP_START op_030_preprocess_normalize -->
## Background subtraction, negative clipping, peak normalization

## Background subtraction, negative clipping, peak normalization

Prepare the measured signal before fitting:

```python
signal = np.asarray(measurements, dtype=float)
signal = signal - 0.0        # background subtraction (bg=0 for this task)
signal[signal < 0] = 0.0     # clip negatives to zero
mx = signal.max()
if mx > 0:
    signal /= mx             # peak-normalize to [0, 1]
```

The forward model also peak-normalizes its output, so both measured and model spectra are on the same 0..1 scale, which the residual comparison relies on. The provided measurements are already normalized; re-applying this is idempotent and safe. Do not add synthetic noise at inference time (noise injection is only for data generation).
<!-- ORACLE_OP_END op_030_preprocess_normalize -->

<!-- ORACLE_OP_START op_040_constants_tables -->
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
<!-- ORACLE_OP_END op_040_constants_tables -->

<!-- ORACLE_OP_START op_050_line_structure -->
## Dunham term values, line positions, populations, cross-sections

## Dunham term values, line positions, populations, cross-sections

Implement a `LineStrength` helper bound to `mc = molecular_constants[species]` (with `Const_D` from universal constants).

Term values (rovibrational energy, cm^-1). For level `v`, rotational `j`:
```
Bv = Be - alpha_e*(v+0.5) + gamma_e*(v+0.5)**2
Dv = De + beta_e*(v+0.5)
Hv = H0 + He*(v+0.5)
Fv = Bv*j*(j+1) - Dv*j**2*(j+1)**2 + Hv*j**3*(j+1)**3     # rotational
Gv = we*(v+0.5) - wx*(v+0.5)**2 + wy*(v+0.5)**3 + wz*(v+0.5)**4   # vibrational
term_values(v,j,'sum') = Gv + Fv ; 'Gv' returns Gv ; 'Fv' returns Fv
```

Line position for a transition (v->v+1, j->j+branch):
```
line_pos(v,j,branch) = term_values(v+1, j+branch) - term_values(v, j)
```

Intensity correction `int_corr(j, branch)` returns `(pt, cd)` (Placzek-Teller factor pt, centrifugal-distortion factor cd):
- Q branch (branch=0): `pt = j*(j+1)/((2j-1)*(2j+3))`, `cd = 1 - 6*Be**2/we**2 * j*(j+1)`.
- O branch (branch=-2): `pt = 3*j*(j-1)/(2*(2j+1)*(2j-1))`, `cd = (1 + 4*Be/we*mu*(2j-1))**2`.
- S branch (branch=+2): `pt = 3*(j+1)*(j+2)/(2*(2j+1)*(2j+3))`, `cd = (1 - 4*Be/we*mu*(2j+3))**2`.

Population difference factor `pop_factor(T,v,j,branch)` (Boltzmann, hc/k=1.44):
```
rho_v(v_) = exp(-1.44/(T)* Gv(v_))                    # vibrational (del_Tv=0)
gj = 3*(2 + (-1)*(j_%2))                                # nuclear spin degeneracy for N2 (even j:9, odd j:3)
rho_r(v_,j_) = (2*j_+1)*gj*exp(-1.44/T * Fv(v_,j_))     # rotational
Qv = sum_{v_=0..19} rho_v(v_);  Qr = sum_{j_=0..99} rho_r(v,j_)
f_low = rho_v(v)*rho_r(v,j)/(Qv*Qr)
f_up  = rho_v(v+1)*rho_r(v+1,j+branch)/(Qv*Qr)
pop_factor = f_low - f_up
```

Doppler linewidth (available helper): `doppler_lw(T, nu_0=2300.) = nu_0*(T/MW)**0.5*Const_D`.

These quantities feed the susceptibility sum; keep the exact `1.44`, degeneracy `gj`, partition sums (v:0..19, j:0..99), and branch coefficients.
<!-- ORACLE_OP_END op_050_line_structure -->

<!-- ORACLE_OP_START op_060_relaxation_matrix -->
## Modified-exponential-gap relaxation (G) matrix

## Modified-exponential-gap relaxation (G) matrix

Build a `js = 30` rotational-state collisional relaxation matrix `gamma_mat` (shape 30x30) using the four fit parameters `[alpha, beta, sigma, m]` from `molecular_constants[species].fit_param_relaxation_matrix`. This models rotational energy transfer that causes collisional narrowing of the Q-branch at ~10 bar; the isolated-line approximation cannot reproduce it.

Let `Ej = term_values(0, arange(js), 'Fv')` be rotational energies. For each pair `i<j` with `abs(i-j)%2==0` (even |Δj|):
```
del_E = Ej[j] - Ej[i]
term_1 = (1 - exp(-m)) / (1 - exp(-m*T/295)) * (295/T)**0.5
term_2 = ((1 + 1.5*1.44*Ej[i]/T/sigma) / (1 + 1.5*1.44*Ej[i]/T))**2
gamma_ji = alpha * P/1.01325 * term_1 * term_2 * exp(-beta*del_E*1.44/T)
gamma_ij = gamma_ji * (2*i+1)/(2*j+1) * exp(del_E*1.44/T)   # detailed balance
gamma_mat[j,i] = gamma_ji ; gamma_mat[i,j] = gamma_ij
```
Then enforce column sum rule on the diagonal:
```
for i in range(js): gamma_mat[i,i] = -sum(gamma_mat[:,i])
```

Key constants: reference pressure `P0 = 1.01325` (atm, in the `P/1.01325` factor), reference temperature `295` K, the `1.5` prefactor, and `hc/k=1.44`. `P` is in bar from `pressure_bar`. This matches the Martinsson et al. 1993 G-matrix form.
<!-- ORACLE_OP_END op_060_relaxation_matrix -->

<!-- ORACLE_OP_START op_070_susceptibility -->
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
<!-- ORACLE_OP_END op_070_susceptibility -->

<!-- ORACLE_OP_START op_080_nonresonant_intensity -->
## Non-resonant background and intensity assembly

## Non-resonant background and intensity assembly

Add the coherently-interfering non-resonant susceptibility and form intensity.

```
chi_nrs_dict = CHI_NRS['SET 1']
chi_val = chi_nrs_dict['SPECIES']['N2'] * x_mol
chi_nrs_eff = chi_val * (P/chi_nrs_dict['P0'] * chi_nrs_dict['T0']/T) * 1e-18
num_density = P/T * Const_N                      # Const_N = 0.724
chi_total = (x_mol * num_density * chi_rs + chi_nrs_eff) * 1e15
I_as = np.abs(chi_total)**2
```

Constants to preserve exactly: non-resonant scaling factor `1e-18`, overall susceptibility scale `1e15`, ideal-gas-like number density `P/T * Const_N`, and reference `P0=1.01325`, `T0=273.15` from the chi_nrs table. `P` is in bar, `T` in K. The non-resonant term interferes coherently (added to complex chi before squaring), which is essential to the line shape.
<!-- ORACLE_OP_END op_080_nonresonant_intensity -->

<!-- ORACLE_OP_START op_090_convolutions_downsample -->
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
<!-- ORACLE_OP_END op_090_convolutions_downsample -->

<!-- ORACLE_OP_START op_100_inversion_solver -->
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
<!-- ORACLE_OP_END op_100_inversion_solver -->

<!-- ORACLE_OP_START op_110_output_writing -->
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

Keys must be exactly `estimated_temperature_K` (shape `(1,)`), `reconstructed_spectrum` (shape `(200,)`), `nu_axis` (shape `(200,)`). Use float32/float64 only; the grader does not cast. `nu_axis` must equal the input axis. A reusable writer helper is bundled at `${CLAUDE_SKILL_DIR}/scripts/op_110_write_submission.py`.
<!-- ORACLE_OP_END op_110_output_writing -->

<!-- ORACLE_OP_START op_120_metrics_diagnostics -->
## Fit-quality metrics for self-diagnosis

## Fit-quality metrics for self-diagnosis

Optional diagnostics to confirm the fit before writing output (not part of the submission). Compare the best-fit spectrum `y_pred` to the measured spectrum `y_true`:
```
mse  = mean((y_true - y_pred)**2)
psnr = 10*log10(1.0/mse)                      # if mse>0
ncc  = dot(y_true, y_pred)/(||y_true||*||y_pred||)     # cosine similarity, higher is better
drange = y_true.max() - y_true.min()
nrmse = sqrt(mse)/drange                       # if drange>0, lower is better
```
Guidance: a good inversion has high NCC (near 1) and low NRMSE. The reference success criterion also targets temperature error under ~50 K vs truth. Use these only to sanity-check convergence (e.g. re-run or adjust the initial guess if NCC is poor); they do not affect the written NPZ.
<!-- ORACLE_OP_END op_120_metrics_diagnostics -->

<!-- ORACLE_OP_START op_130_smoke_probe -->
## Cheap forward-model and pipeline smoke probe

## Cheap forward-model and pipeline smoke probe

Before running the full inversion, verify the forward model returns a finite, peak-normalized `(200,)` spectrum on the case axis and that a single least-squares step is well-posed. This catches shape/axis/constant-loading errors quickly.

Run the bundled probe from the run root with the run-local interpreter:
```
public/envs/runtime/.venv-posix/bin/python ${CLAUDE_SKILL_DIR}/scripts/op_130_smoke_probe.py case_000
```
Expected: prints spectrum length 200, max ~1.0, all-finite True, and a nonzero residual norm at the initial guess. This probe is non-mutating (writes nothing to `outputs/`).
<!-- ORACLE_OP_END op_130_smoke_probe -->
