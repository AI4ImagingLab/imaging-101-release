# Protein Equilibrium Free Energy Surface

> Reconstruct a protein's unfolding free energy over temperature and denaturant concentration from paired fluorescence curves whose folded and unfolded baselines overlap.
> Domain: biology | Keywords: fluorescence spectroscopy, thermodynamic inference, inverse problems | Difficulty: Hard

## Background

A protein's folded and unfolded conformations expose aromatic residues to different molecular environments. Their fluorescence therefore changes as equilibrium shifts with temperature or chemical denaturant. Measuring two emission wavelengths provides complementary sensitivity to that shift, but neither intensity directly measures a conformation population: each conformation also has its own temperature-dependent optical baseline.

Combining thermal and chemical perturbations helps distinguish free energy, enthalpy, heat-capacity and denaturant effects. Quantitative stability measurements support protein formulation, variant characterization and biochemical studies. Inferring these quantities requires a consistent equilibrium model and careful treatment of baselines and units.

## Problem Description

The observations are paired 330nm/350nm fluorescence curves from 15 acyl-coenzyme A binding protein samples with different guanidinium chloride concentrations. Reconstruct the two-dimensional unfolding free energy and its shared thermodynamic and channel-specific baseline parameters. This is a nonspatial physical-parameter inverse problem: its axes are experimental temperature and concentration.

Let $t$ be temperature in degrees Celsius, $T=t+273.15$ absolute temperature in K, $D$ denaturant concentration in mol/L, and $T_m=t_m+273.15$ the zero-denaturant thermal midpoint. The unfolding free energy in kJ/mol is

```math
G(t,D)=H(1-T/T_m)+C(T-T_m-T\log(T/T_m))-mD.
```

Here $H$ is unfolding enthalpy at the thermal midpoint in kJ/mol, $C$ is the unfolding heat-capacity change in kJ/(mol K), and $m$ is the denaturant dependence in kJ/(mol mol/L). The selected physical orientation has $H,C,m>0$. Higher-order temperature dependence of $m$ is fixed to zero. With $R=0.008314$kJ/(mol K), the unfolded population is

```math
u(t,D)=\frac{\exp[-G(t,D)/(RT)]}{1+\exp[-G(t,D)/(RT)]}.
```

For channel $c$, define $s=t-25$K and folded/unfolded baselines

```math
F_c=a_{F,c}+b_{FD,c}D+b_{FT,c}s,\qquad
U_c=a_{U,c}+b_{UD,c}D+b_{UT,c}s+b_{UT2,c}s^2.
```

The modeled normalized intensity is $J_c=(1-u)F_c+uU_c$. Each observed channel and sample is normalized by its own recorded intensity nearest70°C; this convention removes multiplicative sample intensity scaling and fixes the baseline scale. Observational residuals and model mismatch remain. Baseline/thermodynamic coupling makes the inverse nonlinear; both channels share the same population and all samples share the same thermodynamics.

Save `output/reconstruction.npz` with all three required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 15, 3181) | float64 | $G$ in kJ/mol at every supplied sample/temperature, preserving both native axis orders |
| thermodynamic_parameters | (1, 4) | float64 | Ordered $(t_m,H,C,m)$ with units °C, kJ/mol, kJ/(mol K), kJ/(mol mol/L), respectively |
| baseline_coefficients | (1, 2, 7) | float64 | Channels330nm then350nm; coefficient order $(a_F,b_{FD},b_{FT},a_U,b_{UD},b_{UT},b_{UT2})$; units1, L/mol, K^-1,1,L/mol,K^-1,K^-2 |

Every array is real, finite, and uses batch size1. Evaluate every energy-grid value and all18 free parameters. Thermodynamic differences are normalized componentwise by the larger of one in its stated unit and the reference magnitude; baseline and energy differences use maximum absolute error. Full-intensity relative L2 consistency and agreement of the reported energy with the reported thermodynamic parameters are also required. There is no cropping, fitted rescaling, channel permutation or thermodynamic sign reversal in evaluation. Reference agreement measures reproducibility of the declared equilibrium analysis, not independent accuracy of molecular thermodynamics.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| temperature_celsius | (1, 15, 3181) | float64 | Observed temperature for each sample and frame, °C; ascending within each sample |
| denaturant_molar | (1, 15) | float64 | Native sample-order guanidinium chloride concentrations, mol/L |
| intensity | (1, 2, 15, 3181) | float64 | All original supplied processed fluorescence intensities in author instrument units; axes batch, wavelength, sample, temperature frame |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Protein | Acyl-coenzyme A binding protein | — | Selected complete author acquisition |
| Denaturant | Guanidinium chloride | — | Equilibrium chemical perturbation |
| Channels | 330,350 | nm | Emission channel axis order |
| Temperature range | 15–95 | °C | Full original measurement range |
| Sample order | 3.01,2.81,2.60,2.41,2.21,2.00,1.81,1.60,1.40,1.21,1.00,0.80,0.60,0.40,0.19 | mol/L | Native descending concentration order; exact array values authoritative |
| Normalization temperature | 70 | °C | Divide each individual curve by its own nearest recorded intensity; no interpolation |
| Baseline expansion reference | 25 | °C | Origin of baseline temperature powers |
| Celsius offset | 273.15 | K | Add to Celsius for thermodynamics |
| Gas constant | 0.008314 | kJ/(mol K) | Required energy convention |
| Population model | Two equilibrium states | — | Shared folded/unfolded population across both channels |
| Heat capacity | Constant | kJ/(mol K) | Shared over the measured domain |
| Denaturant slope | Temperature independent | kJ/(mol mol/L) | Both higher-order slope coefficients fixed zero |
| Unknown noise scale | Not independently calibrated | instrument units | Equal-weight normalized-intensity least squares; no calibrated chi-square significance claim |
| Boundaries | Every measured point | — | No extrapolation or discarded sample/frame required |

## Method Hints

Use a global nonlinear least-squares model with a shared equilibrium population and separate optical baseline coefficients. The appropriate objective is the sum over both channels and every sample/temperature of $(J_c^{\mathrm{model}}-J_c^{\mathrm{observed}})^2$, where observed intensities use the declared observation-derived normalization. A general Levenberg–Marquardt method suits the smooth low-dimensional thermodynamic model. Linear or curved observed baseline regions can initialize optical coefficients, while physically oriented thermodynamic guesses avoid the interchangeable mathematical folded/unfolded labeling. Fit both channels jointly and examine residuals and sensitivity; neither a melting-point summary nor a channel ratio alone specifies all required quantities.

## References

Hamborg et al., “Global analysis of protein stability by temperature and chemical denaturation,” *Analytical Biochemistry*605,113863(2020), [doi:10.1016/j.ab.2020.113863](https://doi.org/10.1016/j.ab.2020.113863). Author preprint version1(20April2020): [doi:10.1101/2020.04.20.049429](https://doi.org/10.1101/2020.04.20.049429).
