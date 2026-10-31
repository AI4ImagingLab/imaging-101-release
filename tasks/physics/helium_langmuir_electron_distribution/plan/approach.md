# Implemented method

The task uses all 304 supplied Beckers2017 measured voltage/current samples. There is no new acquisition simulation and no hidden actual plasma truth. The numerical oracle is the native PlasmaPy 2025.8.0 single-Maxwellian swept-probe analysis followed by its native Druyvesteyn distribution calculation. Its observational calibration is only cylindrical probe geometry, gas mass, electron mass, and elementary charge; no fitted quantity is a solver input.

## Conventions and preprocessing

Let $v_j$ and $i_j$ denote voltage in V and signed current in A. Average current at identical voltage values, sort ascending, then retain the full characteristic. This implements upstream Characteristic initialization, including its averaging rule; this particular sweep has 304 unique samples. The exposed area is $A=\pi LD+\pi D^2/4$, with $L=1.145\times10^{-3}$ m and $D=1.57\times10^{-3}$ m. Ion current is negative. Temperature $T$ denotes its numerical value in eV; $eT$ is energy in J. Constants are those listed in README, including the full native ion mass.

## Potentials, ion collection, and temperature

Compute the full-sweep first derivative $g=\mathrm{gradient}(i,v)$ with NumPy's default second-order centered interior/nonuniform spacing formula and first-order endpoint rule. Set the plasma-potential index to the first maximizer of $g$, and floating index to the first minimizer of $|i|$. Their voltage values are $V_P,V_F$. Electron saturation current $I_{es}$ is the measured current at the plasma index; ion saturation $I_{is}$ is the minimum of the entire measured current.

Select all $v<V_F$. Fit $10^6i^2=av+b$ by unweighted degree-one least squares (upstream mA² convention). Extrapolate ion current as $I_i(v)=-10^{-3}\sqrt{\max(av+b,0)}$ A. The truncation inside the square root is the native physical OML support boundary, not clipping of the reconstructed EEDF. Derive OML ion density as

```math
n_{i,\mathrm{OML}}=\sqrt{\frac{-10^{-6}a\pi^2m_i}{2A^2e^3}}.
```

First select $V_F<v<V_P$ and subtract the extrapolated ion current. Remove nonpositive separated electron currents before taking their natural log. Fit $h(v;x,y,T)=y+(v-x)/T$ to log current using `scipy.optimize.curve_fit`, its unbounded default Levenberg–Marquardt method, initial parameters all one, numerical Jacobian, native default stopping and function-evaluation limits. There is no regularizer, weighting, line search, or manually modified stopping rule. The intercept parameters $x,y$ are redundant; compare predicted current and physical temperature rather than these individually.

The first temperature $T_1$ determines a refined, strict window $V_F+1.5T_1<v<V_P-0.2T_1$. Refit the same log model to ion-subtracted positive current in that window. This gives the final temperature $T_e$. Both windows are computed from observations, with native constants 1.5 and 0.2, and remain visible in notebook comparisons. No region is selected using a hidden target. Native electron extrapolation above maximum observed current is masked only in plotting; the mathematical exponential is retained as a finite diagnostic.

Density formulas are the electron thermal saturation and ion Bohm collection expressions defined in README. Electron and ion LM estimates use the same final temperature but are not forced equal. No adjustment is made to reconcile their model-dependent discrepancy.

## Independent acquisition integral

For an isotropic energy distribution $f(E)$ per eV, the retarding electron-current operator is

```math
I_e(V)=\frac{Aen_e}{4}\sqrt{\frac{2e}{m_e}}
\int_{U}^{\infty}\frac{E-U}{\sqrt{E}}f(E)\,dE,
\qquad U=\max(V_P-V,0).
```

Here $U$ is the barrier energy in numerical eV and every other quantity has the README units. `distribution_current` evaluates this kernel by trapezoidal quadrature on a supplied positive ascending energy grid. The independent Maxwellian case $f(E)=2\sqrt{E}\exp(-E/T)/(\sqrt{\pi}T^{3/2})$ reproduces the thermal saturation/exponential current formula without calling the inverse. This verifies the square-root factor and charge/mass/area scale. The benchmark inverse preserves the native differential estimate instead of fitting an unmeasured high-energy tail; the forward helper never synthesizes or replaces observations.

## Distribution inversion

Compute $q=\mathrm{gradient}(g,v)$ on all 304 points, including points outside the selected distribution interval. Only then select $V_F<v<V_P$ and define $E=V_P-v$ eV; it is a descending 64-point energy axis. Set $p=q\sqrt{E}$, $Z=|\mathrm{trapezoid}(p,E)|$, and $f=p/Z$. Do not clip negative curvature or change interval endpoints. The particular first helium sweep has positive $f$; perturbed data may not. Normalization here is over a measured finite energy interval, not an estimate of the unobserved electron tail. The differentiation corresponds to the normalized Druyvesteyn integral inverse; constants cancel under normalization. An analytic quadratic-current case checks curvature sign, the square-root energy factor, endpoint ordering, and normalization independently of upstream fixtures.

## Evaluation and experiments

All ten declared outputs are evaluated. Distribution error is L2-relative to the independently frozen original method; energy error is max absolute eV error; scalar error is the maximum absolute relative error over eight scalars. An independent standalone Judge also requires unit absolute trapezoidal integral. Nonzero numerical boundaries were frozen before extraction: 1e-7 distribution, 1e-10 eV axis, 1e-6 scalar, 1e-7 normalization. Exact upstream repeat and local agreement justify numerical tolerances; they do not establish physical accuracy. Zero, changed-scale, wrong-sign, and altered scalar submissions are rejected.

The notebook actively runs the full cleaned inverse, overlays native upstream outputs, compares first/refined fit windows, an independently fitted single-window temperature, and the ideal Maxwellian distribution predicted by the recovered temperature. It computes current residuals only where the retarding model applies and separately displays all observed currents. Noise perturbations and voltage subsampling are sensitivity studies, not new benchmark observations or confidence intervals. No additional smoothing is applied to the benchmark path.
