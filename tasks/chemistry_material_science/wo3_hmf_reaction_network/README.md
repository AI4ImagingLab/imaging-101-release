# WO3 HMF Reaction Network

> Recover a branched photochemical reaction network and its unobserved products from repeated chromatographic concentration measurements and accumulated electrical charge.
> Domain: chemistry & material science | Keywords: reaction networks, electrochemical inference, kinetic inversion | Difficulty: Hard

## Background

Photoelectrochemical oxidation uses absorbed sunlight to generate charge carriers in a semiconductor electrode. Oxidizing holes can convert biomass-derived molecules into useful chemical feedstocks. The electrical charge reports electron transfer, while chromatography measures selected molecular products.

Competing oxidation and degradation pathways make selectivity difficult to infer from a single endpoint. Combining molecular assays with continuous electrical measurements constrains pathways leading to unobserved degradation products. Such analysis supports catalyst assessment and the production of renewable monomers.

## Problem Description

The observations monitor 5-hydroxymethylfurfural (HMF) oxidation on a tungsten trioxide photoanode. Infer eleven nonnegative first-order reaction rates and the fifteen-state concentration evolution. Define the five tracked species in this order: HMF, DFF (2,5-diformylfuran), HMFCA (5-hydroxymethyl-2-furancarboxylic acid), FFCA (5-formyl-2-furancarboxylic acid), and FDCA (2,5-furandicarboxylic acid). Their concentrations form the vector u. Five corresponding first degradation pools form v, and five further degradation pools form w. The complete state is c=(u,v,w), in micromol/L.

The network has HMF→DFF at rate a, HMF→HMFCA at b, DFF→FFCA at d, HMFCA→FFCA at e, and FFCA→FDCA at f. The five u species degrade into the corresponding v pools at rates g,h,i,j,k, respectively. Every v pool converts to its corresponding w pool at common rate l. All rates are in inverse hours. Thus

```math
\begin{aligned}
\dot u_0&=-(a+b+g)u_0,&\dot u_1&=a u_0-(d+h)u_1,\\
\dot u_2&=b u_0-(e+i)u_2,&\dot u_3&=d u_1+e u_2-(f+j)u_3,\\
\dot u_4&=f u_3-k u_4,&\dot v_r&=s_r u_r-lv_r,\quad \dot w_r=lv_r,
\end{aligned}
```

Here r=0,...,4 and s=(g,h,i,j,k). Time is in hours. Initially u equals the mean initial chromatographic assay, and v=w=0. This is a model assumption about unobserved degradation pools. The electrical observation is Q(t)=F V 10⁻⁶ sum(n(m)[c(m,t)−c(m,0)]), where F is the Faraday constant in C/mol, V the solution volume in L and m=0,...,14 the state index. For the five measured species the electron equivalents relative to HMF are n=(0,2,2,4,6); every known oxidation arrow transfers two electrons. The first and final unobserved pools are assigned respectively two and four additional electron equivalents relative to their corresponding parent. These unknown-pool increments are phenomenological assumptions because the byproducts are not chemically characterized. Charge is referenced to the initial measured mixture. The measured concentrations and charge include experimental variability and model discrepancy.

Save **output/reconstruction.npz**, with every following real float64 array:

| Output key | Shape | Units and axis order |
|---|---|---|
| rates | (1, 11) | h⁻¹; batch, (a,b,d,e,f,g,h,i,j,k,l) |
| reconstruction | (1, 150, 15) | micromol/L; batch, time, state (u then v then w) |
| charge | (1, 150) | C; batch, time |

The output time grid consists of 150 equally spaced points including 0 and 64 hours. Preserve absolute concentration and charge scale; no alignment or rescaling is performed. Evaluate each state trajectory and charge by RMSE divided by that reference trajectory's range. Each rate is evaluated separately with an additive plus relative numerical tolerance. Agreement is with the specified standard kinetic method and does not establish physical truth. In particular rate i, the HMFCA degradation pathway, is weakly identifiable near zero; do not interpret a fitted zero as evidence that this pathway is absent.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| concentration_repeats | (1, 3, 30, 6) | float64 | Three author HPLC repeats; last axis time (h), then five u concentrations (micromol/L) |
| charge_repeats | (1, 3, 3000, 2) | float64 | Three electrical repeats; last axis time (h), cumulative charge (C) |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Photoanode | WO3 | material | Tungsten trioxide photoelectrode |
| Applied potential | 0.68 | V versus RHE | Constant photo-oxidation potential |
| Illumination | 3 | equivalent Suns | Simulated solar illumination for prolonged oxidation |
| Electrode area | approximately 3 | cm² | Larger-area electrode used for product monitoring |
| Cell | Two compartments separated by Nafion | geometry | Prevents cathodic reduction of oxidation products |
| Buffer | Aqueous sodium phosphate | electrolyte | Oxidation medium |
| Electrolyte pH | 4 | dimensionless | Aqueous oxidation experiment |
| Nominal initial HMF | 5 | mmol/L | Actual initial assay values are in the observations |
| Solution volume | 0.100 | L | Charge conversion volume |
| Faraday constant | 96485.33212331001 | C/mol | Electron charge times Avogadro constant |
| Assay time extent | 0 to 65 | h | Preserve all 30 rows in fitting |
| Electrical time extent | 0 to 64 | h | Preserve all 3000 rows in fitting |
| Repeats | 3 | acquisitions | Average same-index rows as in author analysis |
| Noise | Uncalibrated | concentration/charge units | Repeated acquisitions support variability estimates; no known Gaussian variance is asserted |
| Boundary/compartment | Closed well-mixed fifteen-state network | model assumption | Total modeled molecular concentration conserved |
| Initial unmeasured pools | 0 | micromol/L | Assumed initial degradation-product concentrations |

The selected reaction-network method explicitly corrects an electron-count indexing error in its published implementation and references acquired charge to the initial state. It retains the original kinetic model and objective, with multiple independent starts to avoid poor local minima; this is a disclosed correction and extension.

## Method Hints

Joint nonlinear kinetic inversion integrates the mass-action equations and fits the chemical and electrical observations together. Nonnegative rate constraints enforce the reaction interpretation. A useful dimensionless residual for a measured value y and modeled value z is (y-z)/(y+z), defined as zero when both vanish. This balances highly unequal concentration scales. The electrical term helps constrain unmeasured pools; its larger sampling density must be retained when reproducing the selected method. Assess residuals and parameter sensitivity alongside fitted rates.

## References

1. Lhermitte, Plainpan, Canjura, Boudoire and Sivula. *Direct photoelectrochemical oxidation of hydroxymethylfurfural on tungsten trioxide photoanodes*. RSC Advances 11, 198–202 (2021). [DOI:10.1039/D0RA09989A](https://doi.org/10.1039/D0RA09989A).
2. Boudoire. *chemical-kinetics: Advanced example, HMF oxidation on WO3*. Author kinetic analysis and experimental observations, MIT license.
