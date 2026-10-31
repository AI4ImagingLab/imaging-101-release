# Helium Langmuir Electron Distribution

> Reconstruct an electron energy distribution and plasma parameters from a measured cylindrical-probe current–voltage sweep despite differentiation noise and overlapping electron and ion currents.
> Domain: physics | Keywords: plasma diagnostics, energy distribution, inverse problems | Difficulty: Hard

## Background

A small biased electrode immersed in plasma collects charged particles. Sweeping its voltage changes the electrostatic energy barrier seen by the electrons. The resulting current depends on their velocity distribution and on the ions collected through the surrounding sheath. Probe measurements give access to conditions inside laboratory discharges and are useful for characterizing plasma processing, fusion experiments, and low-temperature plasmas.

The measured current mixes species whose collection mechanisms differ. Low-energy electrons are strongly suppressed at negative bias, while ions can still reach the probe. The transition into electron collection carries information about the electron energy distribution, but recovering that information amplifies measurement noise. Temperature and density estimates additionally depend on probe geometry and collection-model assumptions.

## Problem Description

Analyze the complete 304-sample single-probe helium characteristic supplied here. Reconstruct its normalized electron energy distribution and the plasma quantities specified below. The characteristic has already received the smoothing supplied with the measurement; its unprocessed precursor is unavailable. Treat the supplied characteristic as the observation.

The current is the sum of ion and electron collection: $I(V)=I_i(V)+I_e(V)$, where $V$ is bias in volts, $I$ is signed current in amperes, and subscripts $i,e$ denote ions and electrons. Ion current is negative. In the retarding region a Maxwellian electron model gives

```math
I_e(V)=I_{es}\exp\left(\frac{V-V_P}{T_e}\right),\qquad
I_i(V)^2=-\frac{2 A^2 e^3 n_{i,\mathrm{OML}}^2}{\pi^2 m_i}(V-V_0).
```

Here $V_P$ is the plasma potential, $T_e$ is electron temperature expressed in eV so its numerical value is a voltage scale, $I_{es}$ is electron saturation current, $A$ is exposed probe area, $e$ is the elementary charge, $m_i$ is ion mass, $n_{i,\mathrm{OML}}$ is the orbital-motion-limited ion density, and $V_0$ is the fitted ion-current intercept voltage. The second relation applies in the ion-dominated region and does not impose a common electron and ion density estimate.

More generally, a normalized Druyvesteyn distribution satisfies

```math
f(E)=C\sqrt{E}\frac{d^2I}{dV^2},\qquad E=V_P-V,\qquad
\int f(E)\,dE=1.
```

The numerical electron energy $E$ is in eV, $f$ is in inverse eV, and $C$ normalizes the integral over the measured floating-to-plasma interval. The floating potential $V_F$ is the bias with zero net current. This approximation neglects ion-current curvature when differentiating the total measured characteristic. Differentiation is ill-conditioned; fitted current separation and the finite measured energy interval limit interpretation. The two density estimates from saturation currents are defined by

```math
n_e=\frac{4I_{es}}{Ae\sqrt{8eT_e/(\pi m_e)}},\qquad
n_i=\frac{|I_{is}|}{0.6Ae\sqrt{eT_e/m_i}},
```

where $m_e$ is electron mass and $I_{is}$ is signed ion saturation current. The factor $eT_e$ converts the numerical temperature in eV into joules. These estimates assume low ion temperature and the collection approximations described above; they need not agree exactly in measured data.

Save **`output/reconstruction.npz`** relative to the task workspace. Every required array must be finite, real `float64`, with batch axis first:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `reconstruction` | `(1, 64)` | float64 | Normalized EEDF, inverse eV; sample order follows descending `energy`, no clipping or rescaling during evaluation |
| `energy` | `(1, 64)` | float64 | Electron energy axis, eV, descending; floating-to-plasma interval endpoints excluded |
| `T_e` | `(1, 1)` | float64 | Electron temperature, eV |
| `n_e` | `(1, 1)` | float64 | Electron saturation-current density estimate, m^-3 |
| `n_i` | `(1, 1)` | float64 | Ion saturation-current density estimate, m^-3 |
| `n_i_OML` | `(1, 1)` | float64 | Orbital-motion-limited ion density estimate, m^-3 |
| `V_P` | `(1, 1)` | float64 | Plasma potential, V |
| `V_F` | `(1, 1)` | float64 | Floating potential, V |
| `I_es` | `(1, 1)` | float64 | Electron saturation current, A, positive |
| `I_is` | `(1, 1)` | float64 | Ion saturation current, A, negative |

Evaluation measures relative L2 error of the distribution, maximum absolute energy-axis error, the largest absolute relative error over all eight scalar quantities, and absolute departure of the trapezoidal distribution integral from one. Every metric must pass. No alignment, energy interpolation, cropping, scale adjustment, or sign correction is performed.

## Data Description

`data/raw_data.npz` contains the full measured sweep and no distribution or fitted parameters.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `bias` | `(1, 304)` | float64 | Applied probe bias relative to laboratory voltage reference, V; full ascending native sweep |
| `current` | `(1, 304)` | float64 | Corresponding signed measured current, A; positive electron collection and negative ion collection |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Probe length | 1.145 | mm | Exposed cylindrical side length |
| Probe diameter | 1.57 | mm | Exposed cylinder diameter |
| Exposed area | pi × length × diameter + pi × diameter² / 4 | m² after converting lengths to m | Side surface plus one circular end |
| Gas | He-4+ | species | Singly ionized helium-4 |
| Ion mass | 6.645568142497802e-27 | kg | Neutral helium-4 mass minus one electron, native convention |
| Electron mass | 9.1093837139e-31 | kg | Electron mass used for thermal current |
| Elementary charge | 1.602176634e-19 | C | Positive charge magnitude |
| Bias range | -30.569354753041267 to 6.286638253272976 | V | Entire supplied sweep; no spatial grid or boundary extension |
| Current sign | electron positive, ion negative | A | Signed laboratory collection convention |
| Plasma regime | low ion temperature and low density | qualitative | Native cylindrical OML analysis assumptions |
| Preprocessing | author-supplied smoothing already applied | none | Smoothing kernel and original unsmoothed trace unavailable |
| Noise calibration | not supplied | A | Do not interpret perturbation experiments as measured uncertainty |

## Method Hints

Separate orbital-motion-limited ion collection by a straight-line fit to squared current on the observed ion region $V<V_F$. The native potential and saturation conventions are

```math
V_P=V_{\arg\max_j (D_VI)_j},\qquad
V_F=V_{\arg\min_j |I_j|},\qquad
I_{es}=I(V_P),\qquad I_{is}=\min_j I_j.
```

Here $j$ indexes the complete ascending voltage/current samples, and $D_V$ is the finite-difference derivative on that full grid: centered three-point nonuniform differences in the interior and first-order one-sided differences at endpoints, as in NumPy gradient with edge_order=1. Resolve index ties by the first sample. The saturation quantities are these observed-current proxies; no ion correction is made to the electron saturation proxy. Average duplicate-voltage currents before differentiation if needed.

Fit the positive ion-subtracted electron current in natural-log space by unweighted inverse-slope temperature estimation. The native two-temperature-fit convention selects strict intervals

```math
S_1=\lbrace V:V_F<V<V_P\rbrace,\qquad
S_2=\lbrace V:V_F+1.5T_1<V<V_P-0.2T_1\rbrace.
```

The first interval gives initial temperature $T_1$ in eV; the second gives the required final $T_e$. The numerical temperatures have voltage scale in these inequalities. The ion-current model is extrapolated from its first squared-current fit for both fits, and only positive separated electron current enters the log fit. A second application of the same full-grid derivative to the complete measured current gives the Druyvesteyn distribution, selecting only $V_F<V<V_P$ afterward and normalizing on the resulting descending energy axis. Preserve the signed second derivative; noise is not a justification to silently clip the distribution.

## References

1. H. M. Mott-Smith and I. Langmuir, “The Theory of Collectors in Gaseous Discharges,” Physical Review 28, 727–763 (1926). https://doi.org/10.1103/PhysRev.28.727
2. M. J. Druyvesteyn, “Der Niedervoltbogen,” Zeitschrift für Physik 64, 781–798 (1930). https://doi.org/10.1007/BF01773007
3. J. Beckers, plasma probe measurements distributed with the PlasmaPy Langmuir analysis example (2017); PlasmaPy Community, PlasmaPy 2025.8.0. 
