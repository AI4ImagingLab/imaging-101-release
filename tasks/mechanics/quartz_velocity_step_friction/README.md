# Contact Friction after a Velocity Step

> Infer rate-and-state contact-friction parameters and internal state from a real wet-quartz velocity-step experiment while accounting for machine stiffness, inertia and transient sliding.
> Domain: mechanics | Keywords: contact mechanics, friction identification, nonlinear dynamics | Difficulty: Hard

## Background

Sliding friction responds to both current slip velocity and the history of a contact. A sudden increase in loading rate can cause an immediate friction rise followed by gradual evolution and damped oscillations. Rate-and-state descriptions represent that memory with an internal state variable whose evolution depends on accumulated slip.

Recovering friction parameters from laboratory records helps characterize granular interfaces and assess when sliding becomes unstable. The loading apparatus matters: its compliance and moving mass couple measured loading motion to the actual contact velocity. Treating a loading-point speed as the instantaneous slider speed can therefore distort inferred contact properties.

## Problem Description

Infer the direct-effect parameter a, evolution parameter b, characteristic slip distance Dc and internal state history of a wet ultrafine quartz-gouge layer during its initial stable velocity increase. The experiment holds60MPa confinement and20MPa pore pressure, giving40MPa effective normal stress. All914 observations in the author-selected1.49–1.7mm displacement region are retained after the documented author drift correction. These are real processed measurements. The exact historical author's manually selected subwindow is unavailable; this task binds the complete stated window and an independently loadpoint-derived transition time.

Let x(t) be slider displacement in m, V=dx/dt slider velocity in m/s, theta(t)>0 a state variable in s, psi=log(Vref theta/Dc), and mu friction coefficient. The stress-based machine balance and slip-state law are

```math
M\dot V=K[X(t)+X_0-x]-\sigma\mu_{reg}(V,\psi)-CV,
\qquad \dot x=V,
\qquad \dot\psi=-\frac{V}{D_c}\left[\log(V/V_{ref})+\psi\right].
```

Here M is moving mass per contact area, K machine stiffness in Pa/m, sigma effective normal stress, C a fixed dashpot coefficient and X0 the initial spring offset. The force term uses regularized friction

```math
\mu_{reg}=a\operatorname{asinh}\!\left[\frac{V}{2V_{ref}}
\exp\!\left(\frac{\mu_0+b\psi}{a}\right)\right],
\qquad \mu=\mu_0+a\log(V/V_{ref})+b\psi.
```

The second expression is the reported positive-velocity friction; the regularization supports the dynamics near reversal. Both agree to numerical precision on this selected positive-velocity experiment. At zero velocity the continuous slip-state derivative is zero. Negative velocities are outside the required stable output trajectory.

The loading velocity is Vl(t)=Vlow+(Vhigh−Vlow){1+tanh[alpha(t−tj)]}/2. Its integral with X(0)=0 is X(t)=Vlow t+(Vhigh−Vlow){logcosh[alpha(t−tj)]−logcosh(−alpha tj)}/(2alpha)+(Vhigh−Vlow)t/2. Here alpha controls the speed transition and tj is its known loadpoint-derived timing. The initial slider speed is Vlow and psi(0)=log(Vref/Vlow); choose X0 to balance initial friction and dashpot stress.

Constrain b−a from the observed steady portions: let muBefore be mean friction through the observation nearest tj, and muAfter the mean of the final51 observations. Set b−a=(muBefore−muAfter)/log(Vhigh/Vlow) and mu0=muBefore+(b−a)log(Vlow/Vref). These are explicit selected steady-state assumptions, not true independently measured material constants.

The observation model compares friction in the **measured loadpoint-displacement coordinate**: solve the prescribed machine motion, then linearly interpolate its friction, velocity and psi against X(t) at each measured displacement. Report theta=(Dc/Vref)exp(psi) after that interpolation. Measurement jitter in displacement does not define a negative prescribed loading speed. Do not substitute measured time for the specified displacement-domain comparison.

Save `output/reconstruction.npz` with all real **float64** arrays:

| Output key | Shape | Units and axis order |
|---|---|---|
| reconstruction | (1,3) | Batch, parameter; order a (dimensionless), b (dimensionless), Dc (m) |
| state | (1,914) | Batch, observation coordinate; positive theta in s, sampled in measured-displacement order |
| slider_velocity | (1,914) | Batch, observation coordinate; positive slider velocity, m/s |
| friction | (1,914) | Batch, observation coordinate; normalized friction coefficient, dimensionless |

Evaluation scores a,b,Dc separately in their physical units, state by maximum absolute error in s, slider velocity by relative L2 error, and friction by maximum absolute difference and measured-data RMS. No parameter rescaling, curve shifting or alignment is allowed. Numerical method references do not establish actual true friction parameters or state.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| time | (1,914) | float64 | Acquisition time relative to selected window start, s |
| loadpoint_displacement | (1,914) | float64 | Measured source loadpoint coordinate relative to selected window start, mm |
| friction | (1,914) | float64 | Real normalized friction after author moving-average drift subtraction and0.694 offset, dimensionless |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Material | Min-U-Sil5 quartz,99.5%SiO2 | — | Ultrafine granular contact layer |
| Median grain diameter | 1.9 | micrometre | Source material distribution |
| Initial layer thickness | approximately1.3 | mm | Direct-shear geometry |
| Sliding area | 50×20 | mm² | Contact area used to define mass per area |
| Confining pressure | 60 | MPa | Wet experiment |
| Pore pressure | 20 | MPa | Deionized water, held constant |
| Effective normal stress sigma | 40e6 | Pa | Confinement minus pore pressure |
| Machine stiffness K | 90.5e9 | Pa/m | Independently measured using a steel specimen |
| Mass per area M | 7798.2 | kg/m² | Moving apparatus mass divided by contact area |
| Vlow | 3e-7 | m/s | Loading speed before step |
| Vhigh and Vref | 3e-6 | m/s | Loading speed after step and friction reference velocity |
| Transition time tj | 24.8 | s | Nearest acquisition to independently fitted loadpoint transition24.824239s |
| Transition alpha | 1 | 1/s | Selected author smooth step convention |
| Dashpot C | 25.52645686341918 | Pa s/m | Small fixed source-model coefficient, not independently measured damping |
| Sampling | approximately10 | Hz | Actual CSV and paper; median step0.1s |
| Nominal friction noise | 0.0004 | dimensionless | Source experimental baseline-noise model |
| Drift processing |500point moving average over1.1–2.5mm, interpolated and subtracted; add0.694 | — | Author observation preprocessing, no target-parameter fit |
| Unobserved initial state | steady sliding at Vlow | — | Selected model initialization |

## Method Hints

Use a stiff dynamical integrator for the coupled slider and state equations. Fit log(a) and log(Dc) to keep positive scales and resolve small characteristic slip distances, with b−a determined by the stated steady portions. Use a bounded trust-region least-squares objective with the complete standardized residual vector. The scalar chi-square interface in the released author script is explicitly corrected to implement the stated nonlinear least-squares objective. Inspect loading-coordinate conventions, stable initial equilibrium and sensitivity to initial guesses.

## References

Kim, Lapusta, Faulkner and Bedford, “Bayesian inversion of rate-and-state parameters from stable velocity step experiments on Quartz Gouge close to instability,” Geophysical Journal International247(2026),ggag362, DOI [10.1093/gji/ggag362](https://doi.org/10.1093/gji/ggag362).

Author wet-observation release accompanying the cited publication, CC-BY4.0. The selected task uses the paper's log-parameter nonlinear least-squares comparison method on the bound wet observation window, with an explicit residual-interface correction and a local extraction of the released Rosenbrock23 integrator. It does not claim a full Bayesian posterior or exact original-paper-window reproduction.
