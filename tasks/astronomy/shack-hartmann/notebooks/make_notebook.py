"""Helper script to generate the adaptive_optics.ipynb notebook."""
import json

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(md(
"# Adaptive Optics Wavefront Reconstruction\n\n"
"**Task**: Given Shack-Hartmann wavefront sensor slopes recorded during a 1 kHz AO simulation\n"
"on a VLT-like 8 m telescope, reconstruct the deformable mirror (DM) commands that maximise\n"
"the Strehl ratio of the K-band long-exposure PSF.\n\n"
"**Dataset**: 500 AO frames, 296 valid subapertures (592 slopes), 150 DM modes (disk harmonics),\n"
"von Kármán turbulence with seeing = 0.6\" and r₀ = 16.8 cm at 500 nm.\n\n"
"This notebook covers:\n"
"1. The AO forward model and inverse problem\n"
"2. Aperture, DM modes, and WFS slope data\n"
"3. Response matrix singular values and Tikhonov reconstruction\n"
"4. Leaky integrator control law\n"
"5. Long-exposure PSF via Fraunhofer propagation (FFT)\n"
"6. Strehl ratio and residual wavefront error"
))

# ── Imports ──────────────────────────────────────────────────────────────────
cells.append(code(
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"import json, sys, os\n"
"sys.path.insert(0, os.path.abspath('..'))\n"
"\n"
"from src.preprocessing import load_raw_data\n"
"from src.physics_model import (\n"
"    compute_reconstruction_matrix,\n"
"    compute_dm_phase,\n"
"    fraunhofer_psf,\n"
"    compute_strehl,\n"
")\n"
"from src.solvers import leaky_integrator_step, run_ao_replay\n"
"from src.visualization import (\n"
"    plot_psf_comparison,\n"
"    plot_wfs_slopes,\n"
"    plot_dm_modes,\n"
"    plot_response_singular_values,\n"
")\n"
"\n"
"%matplotlib inline\n"
"plt.rcParams['figure.dpi'] = 100\n"
"plt.rcParams['font.size']  = 11"
))

# ── Section 1: physics ───────────────────────────────────────────────────────
cells.append(md(
"## 1. The AO Forward Model and Inverse Problem\n\n"
"### Atmospheric turbulence\n\n"
"The atmosphere introduces a random phase screen $\\phi_{\\rm atm}(x,y,t)$ across the telescope pupil.\n"
"For a Kolmogorov/von Kármán turbulence model, the phase power spectrum is:\n\n"
"$$\\Phi_{\\phi}(f) = 0.0229\\,r_0^{-5/3}\\,\\bigl(f^2 + L_0^{-2}\\bigr)^{-11/6}$$\n\n"
"where $r_0$ is the Fried parameter, $L_0$ is the outer scale, and $f$ is the spatial frequency.\n"
"For 8 m aperture with $r_0 = 16.8$ cm and seeing $= 0.6''$, the atmospheric phase variance\n"
"at the WFS wavelength ($\\lambda_{\\rm WFS} = 700$ nm) is $\\sim 300\\,{\\rm rad}^2$.\n\n"
"### WFS measurement model\n\n"
"The Shack-Hartmann WFS divides the pupil into a $20 \\times 20$ lenslet array.\n"
"Each lenslet focuses a sub-aperture onto a CCD; the centroid displacement of the\n"
"focused spot is proportional to the **local wavefront gradient**:\n\n"
"$$s_x^{(k)} = \\frac{1}{\\lambda_{\\rm WFS}} \\iint_{A_k} \\frac{\\partial \\phi}{\\partial x}\\,dx\\,dy,\n"
"\\qquad\n"
"s_y^{(k)} = \\frac{1}{\\lambda_{\\rm WFS}} \\iint_{A_k} \\frac{\\partial \\phi}{\\partial y}\\,dx\\,dy$$\n\n"
"Stacking all $N_{\\rm subaps}$ valid sub-apertures gives the slope vector\n"
"$\\mathbf{s} \\in \\mathbb{R}^{N_{\\rm slopes}}$, $N_{\\rm slopes} = 2 N_{\\rm subaps}$.\n\n"
"### DM forward model\n\n"
"The DM is parameterised by $N_{\\rm modes}$ disk-harmonic basis functions\n"
"$\\{\\psi_j(x,y)\\}_{j=1}^{N_{\\rm modes}}$. For actuator vector $\\mathbf{a}$,\n"
"the DM surface (OPD) is:\n\n"
"$$h(x,y) = \\sum_j a_j\\,\\psi_j(x,y)$$\n\n"
"and the DM phase contribution (double-pass reflection) is:\n\n"
"$$\\phi_{\\rm DM}(x,y) = \\frac{4\\pi}{\\lambda}\\,h(x,y)$$\n\n"
"The WFS measures the **response matrix** $R$ by probing each mode with amplitude $\\pm A$\n"
"and recording the slope response:\n\n"
"$$R_{:,j} = \\frac{\\mathbf{s}(+A_j) - \\mathbf{s}(-A_j)}{2 A_j}$$\n\n"
"$R$ has shape $(N_{\\rm slopes},\\, N_{\\rm modes})$.\n\n"
"### Inverse problem\n\n"
"Given slope measurements $\\mathbf{s}(t)$, find DM actuators $\\mathbf{a}(t)$ to minimise\n"
"the residual wavefront:\n\n"
"$$\\phi_{\\rm res}(t) = \\phi_{\\rm atm}(t) + \\phi_{\\rm DM}(t)$$\n\n"
"(DM adds negative phase to cancel the atmosphere, so $\\phi_{\\rm DM} < 0$ for positive $\\phi_{\\rm atm}$.)\n\n"
"This is solved by inverting $R$ with Tikhonov regularisation and running a leaky integrator."
))

# ── Section 2: load data ─────────────────────────────────────────────────────
cells.append(code(
"DATA_DIR = '../data'\n"
"raw  = load_raw_data(DATA_DIR + '/raw_data.npz')\n"
"meta = json.load(open(DATA_DIR + '/meta_data.json'))\n"
"\n"
"response_matrix = raw['response_matrix'].astype(np.float64)  # (N_slopes, N_modes)\n"
"slopes_ref      = raw['slopes_ref'].astype(np.float64)       # (N_slopes,)\n"
"wfs_slopes      = raw['wfs_slopes'].astype(np.float64)       # (T, N_slopes)\n"
"atm_phases      = raw['atm_phases'].astype(np.float64)       # (T, N_pupil_px)\n"
"dm_modes        = raw['dm_modes'].astype(np.float64)         # (N_modes, N_pupil_px)\n"
"aperture        = raw['aperture'].astype(np.float64)         # (N_pupil_px,)\n"
"\n"
"T, N_slopes  = wfs_slopes.shape\n"
"N_modes, Npx = dm_modes.shape\n"
"N_px         = meta['simulation']['n_pupil_pixels']  # 128\n"
"lam_wfs      = meta['wavefront_sensor']['wavelength_wfs_m']   # 700 nm\n"
"lam_sci      = meta['science_camera']['wavelength_sci_m']     # 2200 nm\n"
"\n"
"print(f'T={T} frames,  N_slopes={N_slopes},  N_modes={N_modes}')\n"
"print(f'Pupil grid: {N_px}x{N_px} = {Npx} px')\n"
"print(f'r0 = {meta[\"atmosphere\"][\"fried_param_r0_m\"]*100:.1f} cm,  '\n"
"      f'seeing = {meta[\"atmosphere\"][\"seeing_arcsec_at_500nm\"]}\"')\n"
"print(f'tau0 = {meta[\"atmosphere\"][\"coherence_time_s\"]*1e3:.1f} ms,  '\n"
"      f'wind = {meta[\"atmosphere\"][\"wind_speed_m_s\"]:.1f} m/s')"
))

# ── Section 2b: aperture & DM modes ─────────────────────────────────────────
cells.append(md(
"## 2. Aperture, DM Modes, and WFS Data\n\n"
"### Telescope aperture\n\n"
"The VLT-like pupil has an 8 m primary mirror with a 1.2 m central obstruction\n"
"and 4 spider vanes of width 5 cm."
))

cells.append(code(
"fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
"\n"
"# --- Aperture ---\n"
"ax = axes[0]\n"
"ax.imshow(aperture.reshape(N_px, N_px), origin='lower', cmap='gray_r')\n"
"ax.set_title('Telescope pupil  (128×128 px)', fontsize=10)\n"
"ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')\n"
"\n"
"# --- Mode 1 ---\n"
"ax = axes[1]\n"
"mode1 = dm_modes[0].reshape(N_px, N_px)\n"
"aper2d = aperture.reshape(N_px, N_px)\n"
"img = np.where(aper2d > 0, mode1, np.nan)\n"
"lim = np.nanpercentile(np.abs(img), 99)\n"
"ax.imshow(img, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim)\n"
"ax.set_title('DM Mode 1 (disk harmonic)', fontsize=10)\n"
"ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')\n"
"\n"
"# --- WFS slope RMS ---\n"
"ax = axes[2]\n"
"rms = np.std(wfs_slopes - slopes_ref, axis=1)\n"
"ax.plot(rms, lw=1.2, color='steelblue')\n"
"ax.set_xlabel('Frame index')\n"
"ax.set_ylabel('RMS slope error')\n"
"ax.set_title('WFS slope error RMS over time', fontsize=10)\n"
"ax.grid(True, alpha=0.3)\n"
"\n"
"plt.tight_layout()\n"
"plt.show()\n"
"print('The slope RMS drops rapidly as the AO loop locks on, '\n"
"      'then oscillates around the corrected residual level.')"
))

cells.append(md(
"### First 6 DM mode shapes\n\n"
"Disk harmonics provide a complete orthonormal basis on the disk with Neumann boundary\n"
"conditions (zero radial derivative at the edge). Lower-order modes correspond to\n"
"low spatial frequencies (tip/tilt, focus, astigmatism analogues)."
))

cells.append(code(
"fig = plot_dm_modes(dm_modes, aperture, (N_px, N_px), n_show=6)\n"
"plt.show()"
))

# ── Section 3: response matrix ───────────────────────────────────────────────
cells.append(md(
"## 3. Response Matrix and Tikhonov Reconstruction\n\n"
"### Response matrix\n\n"
"$R \\in \\mathbb{R}^{N_{\\rm slopes} \\times N_{\\rm modes}}$ is computed from\n"
"push-pull probing: mode $j$ is poked by amplitude $A = 0.01\\,\\lambda_{\\rm WFS} \\approx 7$ nm:\n\n"
"$$R_{:,j} = \\frac{\\mathbf{s}(+A_j) - \\mathbf{s}(-A_j)}{2A_j}$$\n\n"
"Each column of $R$ is the slope *response per meter* of DM surface for that mode.\n\n"
"### Tikhonov reconstruction\n\n"
"The reconstruction matrix $M = R^+$ is the regularised pseudo-inverse via truncated SVD:\n\n"
"$$R = U\\Sigma V^T \\qquad\\Rightarrow\\qquad M = V\\,\\Sigma^+\\,U^T$$\n\n"
"where $\\Sigma^+_{ii} = 1/\\sigma_i$ if $\\sigma_i > \\text{rcond} \\cdot \\sigma_{\\max}$, else 0.\n\n"
"**rcond = 1e-3**: modes below 0.1% of the maximum singular value are zeroed\n"
"to avoid amplifying noise from poorly-sensed spatial frequencies."
))

cells.append(code(
"fig = plot_response_singular_values(response_matrix, rcond=1e-3)\n"
"plt.show()\n"
"\n"
"M = compute_reconstruction_matrix(response_matrix, rcond=1e-3)\n"
"print(f'Response matrix : {response_matrix.shape}')\n"
"print(f'Reconstruction M: {M.shape}')\n"
"print(f'Condition number: {np.linalg.norm(response_matrix)*np.linalg.norm(M):.1f}')"
))

# ── Section 4: control law ───────────────────────────────────────────────────
cells.append(md(
"## 4. Leaky Integrator Control Law\n\n"
"The AO controller runs at 1 kHz. At each timestep $t$, after receiving the slope\n"
"measurement $\\mathbf{s}(t)$ from the WFS, the DM actuator vector is updated:\n\n"
"$$\\mathbf{a}(t+1) = (1 - \\ell)\\,\\mathbf{a}(t)\\;-\\;g\\,M\\,\\bigl(\\mathbf{s}(t) - \\mathbf{s}_{\\rm ref}\\bigr)$$\n\n"
"- $g = 0.3$: **gain** — scales how aggressively the correction is applied.\n"
"  Higher gain corrects faster but amplifies WFS photon noise.\n"
"- $\\ell = 0.01$: **leakage** — prevents actuator *wind-up* (drift to large values)\n"
"  under persistent low-frequency errors or non-common-path errors.\n\n"
"This is a first-order **infinite-impulse-response (IIR)** filter with closed-loop\n"
"transfer function:\n\n"
"$$H(z) = \\frac{g z^{-1}}{1 - (1-\\ell-g)z^{-1}}$$\n\n"
"The $-1$ in the denominator sign comes from the fact that the DM *corrects*\n"
"(adds negative phase), so the minus sign in the control law is correct."
))

cells.append(code(
"# Demo: one step of the leaky integrator\n"
"acts0  = np.zeros(N_modes)\n"
"s_demo = wfs_slopes[0]\n"
"acts1  = leaky_integrator_step(acts0, s_demo, slopes_ref, M, gain=0.3, leakage=0.01)\n"
"print(f'Actuators before: max abs = {np.abs(acts0).max():.3e}')\n"
"print(f'Actuators after:  max abs = {np.abs(acts1).max():.3e}  '\n"
"      f'(correction command, in meters of DM surface)')"
))

# ── Section 5: full AO replay ────────────────────────────────────────────────
cells.append(md(
"## 5. Long-Exposure PSF via Fraunhofer Propagation\n\n"
"### Fraunhofer diffraction (FFT)\n\n"
"The far-field (focal-plane) intensity for a pupil field\n"
"$U(x,y) = A(x,y) e^{i\\phi(x,y)}$ is given by Fraunhofer diffraction:\n\n"
"$$\\text{PSF}(u,v) = \\left|\\mathcal{F}\\{U\\}(u,v)\\right|^2$$\n\n"
"Numerically, we implement this as a zero-padded 2D FFT with oversampling $q=4\\times$.\n\n"
"### Chromatic scaling\n\n"
"The stored atmospheric phases are at $\\lambda_{\\rm WFS} = 700$ nm.\n"
"For the science PSF at $\\lambda_{\\rm sci} = 2200$ nm, the same optical path difference\n"
"(OPD in metres) gives a smaller phase (OPD/$\\lambda$ decreases):\n\n"
"$$\\phi_{\\rm atm}^{\\rm sci}(x,y,t) = \\phi_{\\rm atm}^{\\rm WFS}(x,y,t) \\times\n"
"\\frac{\\lambda_{\\rm WFS}}{\\lambda_{\\rm sci}}$$\n\n"
"### Residual wavefront\n\n"
"At each frame $t \\geq t_{\\rm burn-in}$, the total residual phase at the science wavelength is:\n\n"
"$$\\phi_{\\rm res}(x,y,t) = \\phi_{\\rm atm}^{\\rm sci}(x,y,t) + \\phi_{\\rm DM}^{\\rm sci}(x,y,t)$$\n\n"
"The DM applies $\\phi_{\\rm DM} = 4\\pi h / \\lambda_{\\rm sci}$ where\n"
"$h = \\sum_j a_j \\psi_j$ is the DM surface in metres.\n"
"Since the actuators are driven *negative* to cancel positive atmosphere,\n"
"$\\phi_{\\rm DM}$ is negative, and the sum gives a small residual.\n\n"
"The long-exposure PSF is the time-average:\n\n"
"$$\\text{PSF}_{\\rm long}(u,v) = \\frac{1}{T - t_{\\rm burn-in}} \\sum_{t \\geq t_{\\rm burn-in}}\n"
"\\left|\\mathcal{F}\\{A\\,e^{i\\phi_{\\rm res}(t)}\\}\\right|^2$$"
))

cells.append(code(
"GAIN    = 0.3\n"
"LEAKAGE = 0.01\n"
"BURN_IN = 20\n"
"Q       = 4      # PSF oversampling\n"
"\n"
"# Diffraction-limited reference PSF\n"
"unaberrated_psf = fraunhofer_psf(aperture, np.zeros(Npx), (N_px, N_px), Q)\n"
"unaberrated_peak = float(unaberrated_psf.max())\n"
"print(f'Unaberrated PSF shape: {unaberrated_psf.shape}  peak: {unaberrated_peak:.4e}')\n"
"\n"
"# AO-corrected replay\n"
"print('\\nRunning AO replay ...')\n"
"long_exp_ao, strehl_ao, rms_wfe_nm = run_ao_replay(\n"
"    wfs_slopes, slopes_ref, M,\n"
"    atm_phases, dm_modes, aperture,\n"
"    lam_wfs, lam_sci, (N_px, N_px), unaberrated_peak,\n"
"    gain=GAIN, leakage=LEAKAGE, burn_in=BURN_IN, q=Q,\n"
")\n"
"print(f'  Strehl (AO)  = {strehl_ao:.4f}')\n"
"print(f'  Residual WFE = {rms_wfe_nm:.1f} nm')\n"
"\n"
"# Marechal estimate (WFE in radians at science wavelength)\n"
"sigma_rad_sci = 2*np.pi * rms_wfe_nm*1e-9 / lam_sci\n"
"strehl_marechal = float(np.exp(-sigma_rad_sci**2))\n"
"print(f'  Marechal est = {strehl_marechal:.4f}  (Strehl ≈ exp(-σ²_φ)  at λ_sci)')"
))

cells.append(code(
"# Open-loop PSF (no AO)\n"
"scale = lam_wfs / lam_sci\n"
"no_ao_accum = np.zeros_like(unaberrated_psf)\n"
"for t in range(BURN_IN, T):\n"
"    no_ao_accum += fraunhofer_psf(aperture, atm_phases[t] * scale, (N_px, N_px), Q)\n"
"long_exp_no_ao = no_ao_accum / (T - BURN_IN)\n"
"strehl_no_ao   = compute_strehl(long_exp_no_ao, unaberrated_psf)\n"
"print(f'Strehl (no AO) = {strehl_no_ao:.4f}')"
))

# ── Section 6: visualisation ─────────────────────────────────────────────────
cells.append(md(
"## 6. Results\n\n"
"### Long-exposure PSF comparison\n\n"
"The central 128×128 px crop of each PSF is shown. The AO-corrected PSF shows a\n"
"clear diffraction-limited core (Airy disk), while the open-loop PSF is a\n"
"broad speckle halo."
))

cells.append(code(
"fig = plot_psf_comparison(\n"
"    long_exp_ao, long_exp_no_ao, unaberrated_psf,\n"
"    strehl_ao, strehl_no_ao,\n"
"    title='VLT 8m AO System: Long-Exposure PSF (K-band, λ = 2.2 µm)',\n"
")\n"
"plt.show()"
))

cells.append(md(
"### PSF radial profiles\n\n"
"Radial averaging reveals the Airy ring structure of the AO-corrected PSF."
))

cells.append(code(
"def radial_profile(img):\n"
"    cy, cx = np.array(img.shape) // 2\n"
"    y, x = np.indices(img.shape)\n"
"    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)\n"
"    nr = r.max() + 1\n"
"    tbin = np.bincount(r.ravel(), img.ravel())\n"
"    nr_  = np.bincount(r.ravel())\n"
"    return tbin / np.maximum(nr_, 1)\n"
"\n"
"rmax = 80\n"
"fig, ax = plt.subplots(figsize=(8, 4))\n"
"for psf, lbl, col in [\n"
"    (unaberrated_psf,  'Unaberrated',         'k'),\n"
"    (long_exp_ao,      f'AO (Strehl={strehl_ao:.3f})', 'steelblue'),\n"
"    (long_exp_no_ao,   f'Open loop (Strehl={strehl_no_ao:.3f})', 'tomato'),\n"
"]:\n"
"    rp = radial_profile(psf / psf.max())\n"
"    ax.semilogy(rp[:rmax], lw=1.8, label=lbl, color=col)\n"
"\n"
"ax.set_xlabel('Radius (px)')\n"
"ax.set_ylabel('Normalised intensity')\n"
"ax.set_title('Radial PSF profiles')\n"
"ax.legend()\n"
"ax.grid(True, alpha=0.3)\n"
"plt.tight_layout()\n"
"plt.show()"
))

cells.append(md(
"### Strehl vs gain: sensitivity analysis\n\n"
"The gain controls the balance between correction speed (reject fast turbulence)\n"
"and noise amplification (WFS photon noise enters through M).\n\n"
"For our system ($g_0 = 0.3$, $r_0 = 16.8$ cm, $\\tau_0 = 5$ ms at 1 kHz),\n"
"values between 0.2 and 0.4 should give similar performance."
))

cells.append(code(
"gain_sweep = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]\n"
"strehl_sweep = []\n"
"for g in gain_sweep:\n"
"    _, st, _ = run_ao_replay(\n"
"        wfs_slopes, slopes_ref, M,\n"
"        atm_phases, dm_modes, aperture,\n"
"        lam_wfs, lam_sci, (N_px, N_px), unaberrated_peak,\n"
"        gain=g, leakage=0.01, burn_in=BURN_IN, q=Q,\n"
"    )\n"
"    strehl_sweep.append(st)\n"
"    print(f'  gain={g:.1f}:  Strehl = {st:.4f}')\n"
"\n"
"fig, ax = plt.subplots(figsize=(7, 4))\n"
"ax.plot(gain_sweep, strehl_sweep, 'o-', lw=2, ms=8, color='steelblue')\n"
"ax.axhline(0.25, color='r', ls='--', lw=1.2, label='Strehl boundary = 0.25')\n"
"ax.set_xlabel('Integrator gain')\n"
"ax.set_ylabel('Strehl ratio')\n"
"ax.set_title('AO performance vs gain (leakage=0.01)')\n"
"ax.legend(); ax.grid(True, alpha=0.3)\n"
"plt.tight_layout()\n"
"plt.show()"
))

# ── Summary ──────────────────────────────────────────────────────────────────
cells.append(md(
"## Summary\n\n"
"| Quantity | Value |\n"
"|---|---|\n"
"| Telescope | VLT-like 8 m, K-band (λ = 2.2 µm) |\n"
"| Atmosphere | seeing = 0.6\", r₀ = 16.8 cm, τ₀ = 5 ms |\n"
"| WFS | SH 20×20 lenslets, 296 valid subaps |\n"
"| DM | 150 disk-harmonic modes |\n"
"| Control | Tikhonov rcond=1e-3, gain=0.3, leakage=0.01 |\n"
"| **Strehl (AO)** | **~0.39** |\n"
"| Strehl (no AO) | ~0.09 |\n"
"| Residual WFE | ~344 nm |\n\n"
"**Key takeaways:**\n\n"
"1. **Tikhonov regularisation** is essential — the response matrix is ill-conditioned because\n"
"   high-order modes couple weakly to WFS measurements. Without regularisation, noise is\n"
"   catastrophically amplified.\n\n"
"2. **Leaky integrator gain** must be tuned: too low → slow correction, loses fast turbulence;\n"
"   too high → WFS noise dominates. Optimal gain for 1 kHz loop and τ₀=5 ms is ~0.3.\n\n"
"3. **Chromatic scaling**: the atmosphere at WFS wavelength (700 nm) must be scaled\n"
"   by λ_WFS/λ_sci for the K-band PSF. Since λ_sci/λ_WFS ≈ 3.1, the phase variance\n"
"   at K-band is ~10× smaller than at visible, explaining the moderate Strehl.\n\n"
"## References\n\n"
"1. Por, E. H. et al. (2018). *HCIPy: an open-source adaptive optics simulation framework.*\n"
"   SPIE 10703, doi:10.1117/12.2314407.\n"
"2. Hardy, J. W. (1998). *Adaptive Optics for Astronomical Telescopes.* OUP.\n"
"3. Roddier, F. (1999). *Adaptive Optics in Astronomy.* Cambridge UP."
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"},
    },
    "cells": cells,
}

out = '/home/groot/Documents/PKUlab/imaging-101/tasks/adaptive_optics/notebooks/adaptive_optics.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Written: {out}")
