# Neutron Crystal-Field Hamiltonian

> Reconstruct two competing crystal-field Hamiltonians and temperature-dependent spectral parameters from measured inelastic neutron spectra, then use independently measured magnetic anisotropy to distinguish them.
> Domain: physics | Keywords: neutron spectroscopy, quantum inverse problems, spectral inference | Difficulty: Hard

## Background

The electric environment surrounding a rare-earth ion splits its angular-momentum multiplet into quantum energy levels. Inelastic neutrons exchange energy and angular momentum with these ions, producing peaks whose positions reveal level separations and whose intensities depend on quantum transition matrix elements. This makes neutron spectroscopy useful for investigating magnetic anisotropy and low-energy degrees of freedom in quantum materials.

Powder measurements average over spatial orientations. Different Hamiltonians can consequently produce nearly indistinguishable spectra while predicting very different directional magnetic responses. Measurements at several temperatures provide additional transitions and population information, while independent directional magnetization can help resolve this ambiguity. Approximate backgrounds and broadening also make statistical error bars insufficient to describe all model discrepancies.

## Problem Description

Infer the single-ion crystal-field response of Er3+ in KErSe2 from three measured powder neutron spectra. The known ion has total angular momentum $J=15/2$. In the ascending magnetic basis $m=-J,\ldots,J$, the Hamiltonian is

```math
H=B_{20}O_{20}+B_{40}O_{40}+B_{43}O_{43}+B_{60}O_{60}+B_{63}O_{63}+B_{66}O_{66}.
```

The six real coefficients $B$ have units meV. The dimensionless Stevens operators $O$ are defined below. Let $Z$ be the diagonal angular-momentum operator, $P$ and $Q$ the raising and lowering operators, $X=J(J+1)$, and $I$ the identity. The ladder matrix element is $P_{m+1,m}=\sqrt{(J-m)(J+m+1)}$ and $Q=P^{\mathsf T}$. All other ladder entries vanish. Set $S=P^3+Q^3$ and $A=11Z^3-(3X+59)Z$. Then

```math
\begin{aligned}
O_{20}&=3Z^2-XI,\\
O_{40}&=35Z^4-(30X-25)Z^2+(3X^2-6X)I,\\
O_{43}&=(SZ+ZS)/4,\\
O_{60}&=231Z^6-(315X-735)Z^4+(105X^2-525X+294)Z^2+(-5X^3+40X^2-60X)I,\\
O_{63}&=(SA+AS)/4,\qquad O_{66}=(P^6+Q^6)/2.
\end{aligned}
```

Diagonalizing $H$ gives energies $E_i$ relative to its minimum and normalized states $|i\rangle$. The response at measured energy transfer $e$ and temperature $T$ is

```math
I_T(e)=s\sum_{i:p_i>10^{-3}}\sum_j p_i\sum_{a=x,y,z}|\langle i|J_a|j\rangle|^2
V\bigl(e(1+T\eta)-d-(E_j-E_i);a(E_j-E_i)+b,\gamma_T\bigr)+c_T,
\quad p_i=\frac{\exp[-E_i/(kT)]}{\sum_j\exp[-E_j/(kT)]}.
```

Here $s$ is a common intensity prefactor, $d$ an energy offset, $\eta$ an inverse-temperature axis-expansion coefficient, $c_T$ a constant background and $\gamma_T$ a Lorentzian FWHM for each temperature. The same-data resolution coefficients $a,b$ define Gaussian FWHM as a linear function of transition energy; they must be estimated, not assumed known. The angular momenta are $J_x=(P+Q)/2$, $J_y=(P-Q)/(2\mathrm{i})$, $J_z=Z$, with $\mathrm{i}^2=-1$. For positive Gaussian width, the normalized Voigt profile is

```math
V(u;\alpha,\gamma)=\frac{\Re w((u+\mathrm{i}\gamma/2)/(\sigma\sqrt2))}{\sigma\sqrt{2\pi}},
\quad \sigma=\frac{\alpha}{2\sqrt{2\ln2}},\quad
w(z)=e^{-z^2}\operatorname{erfc}(-\mathrm{i}z).
```

Here erfc denotes the complementary error function. This selected empirical response is defined only on the supplied observation domain. Linear width extrapolation becomes negative for unobserved high-energy levels; its signed source formula has the same far-tail values as the positive-width physical continuation at all bound observations. Do not interpret extrapolated widths or unmeasured high-energy levels as calibrated measurements. Intensities include the processed powder normalization, without an additional final/initial neutron-wavevector factor.

Return both neutron-consistent branches: row 0 is easy-plane and row 1 easy-axis, defined by the predicted physical moments $M_x(1\mathrm{T})>M_z(1\mathrm{T})$ and the reverse at 0.5 K. The moment is $\boldsymbol M=-g\langle\boldsymbol J\rangle$ in Bohr magnetons per Er under $H+g\mu\boldsymbol B\cdot\boldsymbol J$. Here $g$ and $\mu$ are known constants below; the thermal average uses all 16 field-split levels. Independent observed magnetization determines the selected branch. A single-ion model omits exchange interactions and need not reproduce the complete magnetization amplitude exactly.

Save `output/reconstruction.npz` relative to the task workspace with all these numeric arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 2, 15) | float64 | Batch, easy-plane/easy-axis branch, parameter; columns listed below; real physical values |
| neutron_spectrum | (1, 2, 3, 180) | float64 | Batch, branch, temperature, supplied energy index; predicted intensity in author intensity units at every supplied observation |
| resolution_coefficients | (1, 2) | float64 | Batch, linear width coefficients `[a,b]`; respectively dimensionless and meV |
| selected_model | (1,) | int64 | Batch; index 0 or 1 selected using independent directional magnetization |

The 15 columns are `[B20,B40,B43,B60,B63,B66,s,gamma_1.8,gamma_15,gamma_100,d,c_1.8,c_15,c_100,eta]`. Coefficients, three widths and offset use meV; the prefactor uses intensity-unit times meV; three backgrounds use intensity units; expansion uses K^-1. Temperatures are ordered 1.8, 15, 100 K. The common sign change of B43 and B63 is a rotation of the local transverse axes by 60 degrees: parameter comparison canonicalizes row 0 to B43 negative and row 1 to B43 positive, flipping B63 together. No free intensity rescaling, cropping or arbitrary row permutation is allowed.

Evaluation checks every required quantity: all 15 parameters in each branch, both full spectra, inferred width coefficients, physical observation agreement on the declared fit masks, forward consistency and independently justified branch selection. Spectrum NRMSE means RMS difference divided by the comparison spectrum's dynamic range. NCC is cosine similarity without mean subtraction. This is a conditional standard-method comparison, not a claim that neutron-only inversion uniquely identifies a true Hamiltonian.

## Data Description

All observations are author-provided processed experimental data. Intensities and reported uncertainties are on their supplied common author scale. Full spectra are retained; fixed masks identify the author's fitting window and excluded elastic region. Independent magnetization observations retain every finite field/moment pair in source order.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| energy | (1, 3, 180) | float64 | Batch, temperature, energy transfer; meV |
| intensity | (1, 3, 180) | float64 | Measured neutron intensity; author intensity units |
| reported_error | (1, 3, 180) | float64 | Reported intensity error; author intensity units; does not include all model discrepancy |
| temperature | (1, 3) | float64 | Neutron sample temperatures; K |
| fit_mask | (1, 3, 180) | bool | Dimensionless predefined fit inclusion mask; 59, 115, 115 included samples |
| field_ab | (1, 215) | float64 | Applied field parallel to basal plane at 0.5 K; T |
| moment_ab | (1, 215) | float64 | Independent measured moment parallel to basal field; Bohr magnetons per Er |
| field_c | (1, 272) | float64 | Applied field parallel to c axis at 0.5 K; T |
| moment_c | (1, 272) | float64 | Independent measured moment parallel to c field; Bohr magnetons per Er |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Material / ion | KErSe2 / Er3+ | dimensionless | Selected measured sample and known ion |
| J, L, S | 7.5, 6, 1.5 | dimensionless | Ion angular-momentum quantum numbers |
| Local coordinate | z parallel c; x,y transverse | dimensionless | Real Stevens convention; ascending m basis |
| Site symmetry | D3 | dimensionless | Six allowed real Stevens terms in the chosen frame |
| Neutron instrument | HYSPEC | dimensionless | Powder inelastic neutron measurement |
| Incident energy | 9 | meV | Bound measured spectra |
| Chopper frequency | 360 | Hz | Measurement configuration |
| Integrated momentum transfer | 1.25 to 1.9 | inverse angstrom | Powder cut integration interval |
| Sample mass / can diameter | approximately 3 / 9.5 | g / mm | Powder sample in aluminum can |
| Neutron temperature order | 1.8, 15, 100 | K | Source filename 1p6K corresponds to documented 1.8 K acquisition |
| Neutron thermal constant | 0.0861733 | meV/K | Population convention |
| Magnetic temperature / Landé factor | 0.5 / 1.2 | K / dimensionless | Independent directional magnetic comparison |
| Magnetic thermal constant | 0.086173303 | meV/K | Field-response population convention |
| Bohr magneton energy conversion | 0.057883818012 | meV/T | Zeeman energy conversion |
| Source magnetic field conversion | 10000 | Oe/T | Already applied to supplied field arrays |
| Reported noise | supplied positive errors; independence is a fit approximation | intensity units | Errors do not calibrate systematic background or resolution discrepancy |
| Bound energy coordinate | each supplied energy array; no periodic extension | meV | No extrapolation beyond observed domain is evaluated |

## Method Hints

A useful approach estimates empirical peak widths from the lowest-temperature spectrum, then fits a common quantum Hamiltonian to all temperatures with thermal populations and transition amplitudes. Two competing anisotropy branches should be retained rather than treating one powder fit as uniquely identified. Weighted nonlinear optimization can refine the Hamiltonian and nuisance spectral parameters; independent directional magnetic contrast distinguishes the branches. Width estimation is part of the inverse problem and must remain active.

## References

- A. Scheie et al., “Crystal field Hamiltonian and anisotropy in KErSe2 and CsErSe2,” Physical Review B 101, 144432 (2020), [doi:10.1103/PhysRevB.101.144432](https://doi.org/10.1103/PhysRevB.101.144432). Selected analysis is the real three-temperature KErSe2 refinement; the converged computation budget is documented separately from the historical paper budget.
- A. Scheie, “PyCrystalField: software for calculation, analysis and fitting of crystal electric field Hamiltonians,” Journal of Applied Crystallography 54, 356–362 (2021), [doi:10.1107/S160057672001554X](https://doi.org/10.1107/S160057672001554X).
