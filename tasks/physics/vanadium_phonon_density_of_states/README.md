# Vanadium Phonon Density of States

> Reconstruct a normalized vibrational density of states from measured neutron momentum–energy scattering with coupled multiphonon and multiple-scattering contributions.
> Domain: physics | Keywords: neutron spectroscopy, vibrational imaging, spectral inversion | Difficulty: Hard

## Background

Inelastic neutron scattering measures energy and momentum exchanged between neutrons and atomic motion. A vibrational density of states describes the fraction of lattice modes per energy interval. Its shape informs heat capacity, lattice stability and energy transport.

The measured spectrum contains processes involving several phonons and repeated scattering in the sample. These contributions depend on the same unknown density that determines single-phonon scattering. Thermal populations also produce asymmetric energy-gain and energy-loss spectra. Recovering the density therefore requires a coupled physical model.

## Problem Description

The observation is an existing experimental vanadium intensity map. Infer the nonnegative density $g_j=g(jh)$, with $h=1\ \mathrm{meV}$, on energies $j=0,\ldots,79$. It is a fraction of vibrational modes per meV, rather than an absolute count of modes. Its declared physical constraints are

```math
h\sum_{j=0}^{79}g_j=1,\qquad
g_j=0\ (j>55),\qquad
g_j=\frac{j^2}{7^2}g_7\ (0\le j<7).
```

The last condition specifies a Debye low-energy continuation where elastic contamination prevents direct inversion. There is no independently observed density-of-states truth.

The following finite-grid incoherent model defines the conditional inverse target. Define inverse thermal energy $\beta=1/(k_{\mathrm B}T)$. For forward evaluation use the normalized density $g/(h\sum_j g_j)$, while evaluating the submitted normalization constraint separately. Extend this density with 160 zero bins to positive energies $e_j=jh,\ j=0,\ldots,239$. With $b_j=g_j\coth(\beta e_j/2)/e_j$ for $j>0$, set $b_0=2b_1-b_2$ and

```math
\gamma=h\sum_{j=0}^{239}b_j,\qquad
d(Q)=\gamma\frac{\hbar^2 Q^2}{2M}.
```

Here $\gamma$ has units $\mathrm{meV}^{-1}$, the recoil energy is converted to meV, $M$ is atomic mass and $Q$ is momentum-transfer magnitude. On the reflected grid $-239h,\ldots,239h$, define

```math
A_1(E)=\frac{g(|E|)}{\gamma E[1-\exp(-\beta E)]}\quad(E\ne0),\qquad
A_1(0)=\max[0,2A_1(h)-A_1(2h)].
```

For $n=2,3,4$, form a linear convolution $h(A_1*A_{n-1})$, retain its central 479 bins, and normalize its sum times $h$ to one. This defines $A_n$, in $\mathrm{meV}^{-1}$. The original finite-grid definition does **not** normalize $A_1$ separately: its mass has an endpoint quadrature bias of order $h$. All orders obey detailed balance $A_n(-E)=\exp(-\beta E)A_n(E)$.

On the analysis window $E=-15,\ldots,79\ \mathrm{meV}$, define single-phonon intensity $S$, multiphonon intensity $P$, multiple scattering $U$, and total model $F$:

```math
S(Q,E)=d(Q)e^{-d(Q)}A_1(E),\qquad
P(Q,E)=\sum_{n=2}^{4}\frac{d(Q)^n e^{-d(Q)}}{n!}A_n(E).
```

Let $K(Q,E)$ be one for a strict neutron wave-vector triangle and zero otherwise. With incident wave number $k_i$, final wave number $k_f$, and $Q$ in $\mathrm{\mathring A}^{-1}$,

```math
k_i=c\sqrt{E_i},\quad k_f=c\sqrt{E_i-E},\quad
c=\frac{\sqrt{2m_n(10^{-3}q_e)}}{\hbar_n\,10^{10}}.
```

Energies under the square roots are numeric meV values. The factor $c$ is approximately0.6947 in units $\mathrm{\mathring A}^{-1}\mathrm{meV}^{-1/2}$: $10^{-3}q_e$ converts meV to joules and division by $10^{10}$ converts inverse meters to inverse ångströms. The triangle requires $k_i+k_f>Q$, $k_i+Q>k_f$, and $k_f+Q>k_i$. Set $P=0$ outside this triangle. The multiple-scattering approximation is uniform over accessible momentum bins:

```math
U(Q,E)=C\frac{\sum_{Q'}P(Q',E)}{\sum_{Q'}K(Q',E)},\qquad
F(Q,E)=S(Q,E)+P(Q,E)+U(Q,E).
```

Only acquired accessible cells are compared. The sum over $Q'$ includes every supplied momentum bin, with no fitted quality selection.

Resample the native observations as specified below, obtaining $I(Q,E)$. An unknown absolute scattering scale is eliminated by inelastic-area normalization. Let $\Omega(E)$ denote the acquisition mask after resampling:

```math
a(g)=
\frac{\sum_{E\ge7}\sum_{Q\in\Omega(E)}F(Q,E)}
{\sum_{E\ge7}\sum_{Q\in\Omega(E)}I(Q,E)},\qquad
\rho(g)=
\frac{\left\|\left\lbrace\sum_{Q\in\Omega(E)}[a(g)I(Q,E)-F(Q,E)]\right\rbrace_{E=7}^{79}\right\|_2}
{\left\|\left\lbrace\sum_{Q\in\Omega(E)}a(g)I(Q,E)\right\rbrace_{E=7}^{79}\right\|_2}.
```

Recover a density with $\rho\le0.00795285686754709$. Every acquired momentum bin contributes equally. This is a conditional momentum-integrated target: the incoherent approximation leaves substantial structure in individual momentum residuals, and no noise-limited fit of each detector cell is asserted. The coupled nonlinear model and missing observations make direct division inadequate.

Save output/reconstruction.npz containing exactly:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 80) | float32 or float64 | Real $g_j$ in $\mathrm{meV}^{-1}$; batch then increasing energy 0–79 meV; no arbitrary rescaling or alignment |

Every output element must be finite. Evaluation requires the residual bound above, normalization error $\lvert h\sum g-1\rvert\le5\times10^{-6}$, maximum negative part at most $10^{-10}\ \mathrm{meV}^{-1}$, maximum absolute density above 55 meV at most $10^{-9}\ \mathrm{meV}^{-1}$, and maximum low-energy continuation error at most $10^{-7}\ \mathrm{meV}^{-1}$. All conditions apply together. Different physically admissible densities meeting these bounds are valid; a particular iterative trajectory is not required.

## Data Description

The file data/raw_data.npz preserves measured floating-point intensity and variance. A missing detector cell remains NaN and is explicitly identified by the acquisition mask. It is neither a zero measurement nor a hidden value to reconstruct. Variance is retained for uncertainty analysis; the declared momentum-marginal target is unweighted.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| intensity | (1, 300, 149) | float64 | Measured neutron intensity in original arbitrary intensity units; axes batch, momentum, energy; NaN at unmeasured cells |
| variance | (1, 300, 149) | float64 | Variance in squared original intensity units; finite and nonnegative at measured cells |
| measured_mask | (1, 300, 149) | bool | Dimensionless acquisition validity; true exactly at finite intensity cells |
| q_axis | (1, 300) | float64 | Momentum-transfer bin centers in $\mathrm{\mathring A}^{-1}$, increasing |
| energy_axis | (1, 149) | float64 | Original energy-transfer bin centers in meV; positive means neutron energy loss |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Sample | Vanadium experimental scattering map | — | Existing measured data |
| Temperature $T$ | 300 | K | Thermal population |
| Atomic mass | 50.94 | u | Vanadium mass $M$ |
| Incident energy $E_i$ | 120 | meV | Incident neutron energy |
| Momentum centers | 0.025 to 14.975, spacing 0.05 | $\mathrm{\mathring A}^{-1}$ | All 300 bins retained |
| Native energy centers | Given exactly by energy_axis | meV | Original measurement precision retained |
| Analysis energy centers | −15 to 79, spacing 1 | meV | 95-bin finite window |
| DOS grid | 0 to 79, spacing $h=1$ | meV | 80 output bins |
| DOS upper support | 55 | meV | Inclusive physical support |
| Debye continuation | 0 through 6, anchored at 7 | meV | Explicit low-energy condition |
| Multiple-scattering coefficient $C$ | 0.2 | dimensionless | Uniform accessible-momentum approximation |
| Phonon orders | 1 through 4 | dimensionless | Complete selected finite model |
| $k_{\mathrm B}$ | 0.0862 | meV/K | Original numerical thermal convention |
| Recoil $\hbar$ | $1.0545716\times10^{-34}$ | J s | Original recoil convention |
| Atomic mass unit | $1.66053873\times10^{-27}$ | kg/u | Original recoil convention |
| Joule-to-meV factor | $6.2415097\times10^{21}$ | meV/J | Recoil energy conversion |
| Kinematic $m_n$ | $1.6749286\times10^{-27}$ | kg | Neutron mass convention |
| Kinematic $q_e$ | $1.60217733\times10^{-19}$ | J/eV | Energy conversion |
| Kinematic $\hbar_n$ | $1.054571628\times10^{-34}$ | J s | Wave-vector conversion convention |
| Convolution boundary | Zero extension, central retained window | — | 479 reflected bins after positive-grid extension |

For resampling only, replace unmeasured native cells temporarily by zero, then use piecewise bilinear interpolation on the supplied momentum and energy centers. Evaluate at the same momentum centers and the 95 analysis energies. Restore missing energy boundaries separately for each momentum row: let $L,H$ be its first and last measured native energy centers. If $L>-15$, mask all analysis indices up to and including $\mathrm{round}(\min(L,79)+15)$. If $H<79$, mask all indices from $\mathrm{round}(\max(H,-15)+15)$ onward. Round half to even. A row with no observations uses $L=H$ equal to the first native energy center. Apply the identical interpolation and boundary mask to variance. This specifies a finite observation protocol without pretending that missing cells were measured zeros. The resulting acquired cells are inside the neutron triangle.

## Method Hints

An alternating physical correction can update the single-phonon density while recalculating multiphonon scattering, multiple scattering and the Debye–Waller factor from that same density. Positivity, normalization and a low-energy continuation control unstable divisions. Multiplicative methods can trap zeros, so assess the stated physical target rather than assuming that one finite iteration sequence is unique. Inspect both momentum-integrated and momentum-resolved residuals to distinguish reconstruction quality from model error.

## References

1. Lin, Islam and Kresh. *Multiphonon: Phonon Density of States tools for Inelastic Neutron Scattering Powder Data*. Journal of Open Source Software 3, 440 (2018). [doi:10.21105/joss.00440](https://doi.org/10.21105/joss.00440).
2. Kresch et al. *Neutron scattering measurements of phonons in nickel at elevated temperatures*. Physical Review B 75, 104301 (2007). [doi:10.1103/PhysRevB.75.104301](https://doi.org/10.1103/PhysRevB.75.104301). Physical methodology; the bound measurement here is vanadium.
