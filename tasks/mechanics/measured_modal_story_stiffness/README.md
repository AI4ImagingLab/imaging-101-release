# Measured Modal Story Stiffness

> Recover four inter-story stiffnesses and unobserved top-floor modal amplitudes from two measured partial vibration modes despite incomplete sensing and nonconvex mechanical coupling.
> Domain: mechanics | Keywords: inverse dynamics, structural identification, modal imaging | Difficulty: Hard

## Background

A vibrating structure's natural frequencies and displacement patterns depend on its mass and stiffness distribution. Dynamic testing can therefore reveal mechanical properties that cannot be measured directly without disassembling the structure. Different stories influence each vibration mode through coupled force balance.

Accelerometers often cover only accessible floors. Partial modal observations leave unmeasured motion and stiffness correlated, while discrepancies between ideal shear-building models and laboratory structures prevent an exact fit. Recovering stiffness supports condition assessment and predictive vibration modeling.

## Problem Description

The observations are author-processed modes from a real four-story laboratory shear frame mounted on a shake table. Only the first two modes and floors1–3 are supplied. Their identified frequencies and relative mode shapes are observations, not exact physical truth. The fixed-base model has four horizontal floor degrees of freedom and a free top. For mode i,

```math
K(k)u_i=\lambda_i M u_i,\qquad \lambda_i=(2\pi f_i)^2,
\qquad K(k)=\sum_{j=1}^{4}k_j d_jd_j^T.
```

Here f is frequency in Hz, lambda is squared angular frequency in s⁻², M is the diagonal floor mass matrix in lbf·s²/in, k is the vector of inter-story stiffnesses in lbf/in, and u is a dimensionless relative modal displacement. The vectors d1=(1,0,0,0), d2=(-1,1,0,0), d3=(0,-1,1,0), d4=(0,0,-1,1) measure successive story drifts. Each supplied mode is normalized by its signed largest-magnitude measured entry, making that entry+1; use this same normalization for the unknown fourth-floor amplitude. All four masses are fixed and independently weighed.

Infer four stiffnesses and the fourth-floor amplitude of both modes. The admissible model range is0<k≤20lbf/in and absolute unobserved modal amplitude≤3. These are model bounds, not known answers. The data are imperfect and the inverse problem can have local minima. It is not permissible to declare the measured mode vectors to be exact full-structure truth.

Save `output/reconstruction.npz` with:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 4) | float32 or float64 | Real inter-story stiffness in lbf/in; batch then stories1–4 from base to top |
| unmeasured_modes | (1, 1, 2) | float32 or float64 | Real dimensionless relative modal amplitude; batch, unmeasured floor4, ascending observed mode order; normalization fixed above |

Evaluation checks maximum componentwise relative error for both quantities and RMS dynamic-equilibrium residual over the four floors and two modes. For submitted k and latent amplitudes, concatenate each normalized measured three-vector with its submitted fourth entry, then form K(k)u−lambda M u. RMS is in lbf/in under the dimensionless modal convention. There is no gain, sign, permutation, or coordinate alignment. Reference agreement measures reproducibility of a validated standard-method interpretation, not independent knowledge of true stiffness.

## Data Description

`data/raw_data.npz` retains the selected author modal measurements without precision reduction. Raw acceleration time histories are not included in the original example used here.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| frequency | (1, 2) | float64 | Identified frequencies in Hz; batch, ascending mode |
| measured_modes | (1, 3, 2) | float64 | Dimensionless identified modal amplitudes; batch, floors1–3, mode; retain signs and normalize as specified |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Specimen | Laboratory four-story shear frame | dimensionless | Shake-table base excitation |
| Measured floors | 1,2,3 | floor index | Accelerometer-derived modes selected by the author |
| Retained modes | First and second | mode index | Low-frequency observed information |
| Floor weight | 12.060 each | lbf | Independently weighed floors |
| Gravitational acceleration | 386.088 | in/s² | Original mass conversion |
| Floor mass | 12.060/386.088 each | lbf·s²/in | Diagonal mass matrix |
| Boundary conditions | Fixed base, free top | dimensionless | Four lateral shear degrees of freedom |
| Modal scale | Signed measured maximum becomes+1 | dimensionless | Same normalization for measured and latent coordinates |
| Noise covariance | Not supplied | dimensionless | No exact modal uncertainty is asserted |

## Method Hints

Use bounded nonlinear least squares on the modal dynamic residual, jointly updating stiffness and unmeasured modal amplitudes. With nominal stiffness10lbf/in, a relative change alpha gives k=10(1+alpha). The residual is bilinear in stiffness changes and latent amplitudes. Its stiffness derivative is the corresponding influence matrix applied to the complete mode, and its latent derivative is the unmeasured column of K−lambda M. Multiple feasible starts help distinguish good minima from boundary solutions. Modal-frequency/shape discrepancy approaches provide useful comparisons but define different objectives.

## References

Otsuki, Lander, Dong and Wang (2022), “Formulation and application of SMU: An open-source MATLAB package for structural model updating,” Advances in Structural Engineering25(4),698–715. [Published article](https://doi.org/10.1177/13694332211022066).
