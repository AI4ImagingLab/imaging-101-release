# Eclipsing Binary Geometry

> Recover two stellar radii, orbital orientation, eccentricity, infrared brightness ratio, and velocity scale from five photometric time series and radial velocities, with coupled eclipse geometry and light travel time.
> Domain: astronomy | Keywords: stellar occultation, orbital inference, time series | Difficulty: Hard

## Background

When two stars orbit nearly edge on, each can periodically obscure the other. Eclipse depth measures the blocked surface brightness, while eclipse duration and shape constrain stellar size and the path across the projected disk. A secondary eclipse supplies information about the fainter companion that is difficult to obtain from its spectrum alone.

Radial velocities measure orbital motion along the line of sight. Combining these velocities with photometry helps distinguish eccentric orbital motion from changes in stellar radius and viewing angle. Applications include stellar mass and radius measurements and tests of low mass stellar structure. Correlated geometric parameters, different observing bands, and small timing corrections make this a nonlinear inverse problem.

## Problem Description

Infer the seven real parameters of the conditional spherical model of J0113+31. Save `output/reconstruction.npz`, containing exactly the following array:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 7) | float32 or float64 | Ordered parameters: fractional radius of primary, fractional radius of secondary, inclination in degrees, square-root eccentricity sine coordinate, square-root eccentricity cosine coordinate, mean-disk J-band surface brightness ratio, velocity scale in km/s. |

Write this vector as $(r_1,r_2,i,f_s,f_c,S_J,K_1)$. Radii are relative to the semimajor axis of the relative orbit. The companion to primary mass ratio is $q$. Eccentricity and the primary argument of periastron are

```math
e=f_s^2+f_c^2,\qquad \omega=\operatorname{atan2}(f_s,f_c).
```

Use the detached physical domain: both radii are positive, $r_1+r_2<1-e$, $0\le e<1$, $0<i\le90$ degrees, $S_J\ge0$, and $K_1>0$. The canonical inclination removes the mirror orientation ambiguity. There is no requirement to reproduce a particular parameter vector when a different physical vector fits the observations equally well.

The fixed epoch is the exact minimum of sky projected centre separation near primary conjunction. For inclination in radians, define true anomaly $\nu$, eccentric anomaly $E$, periastron epoch $T_p$, and period $P$ by

```math
M=\frac{2\pi(t-T_p)}{P}=E-e\sin E,\qquad
\nu=2\arctan\left[\sqrt{\frac{1+e}{1-e}}\tan\frac E2\right],\qquad R=1-e\cos E.
```

Here $R$ is separation in units of the relative semimajor axis. To obtain $T_p$ from a conjunction epoch, minimize

```math
D(\nu)=\frac{(1-e^2)\sqrt{1-\sin^2i\sin^2(\nu+\omega)}}{1+e\cos\nu}
```

in the interval within $\pi/4$ of $\pi/2-\omega$, then convert the minimizing anomaly to $E$ and $M$. This avoids approximating an inclined eccentric conjunction as exactly $\nu=\pi/2-\omega$.

For infrared photometry and radial velocities, include light travel time. Define the primary orbital semimajor axis $a_1$ and relative semimajor axis $a$, in solar radii, using the specified historical velocity conversion:

```math
a_1=\kappa K_1P\frac{\sqrt{1-e^2}}{\sin i},\qquad
a=(1+q^{-1})a_1,\qquad \ell=\frac{aR_\odot}{86400c}.
```

The conjunction epoch used to determine periastron becomes

```math
T_c=T_0-\ell\sin i\frac{q-1}{q+1}\frac{1-e^2}{1+e\sin\omega}.
```

At observed time $t$, first compute $R,\nu$. With $b_1=q/(1+q)$ and $b_2=1/(1+q)$, the line of sight coordinates are $w_1=-b_1R\sin i\sin(\nu+\omega)$ and $w_2=b_2R\sin i\sin(\nu+\omega)$. Recompute the respective orbital states at $t+\ell w_1$ and $t+\ell w_2$, once, to obtain $(R'_1,\nu'_1)$ and $(R'_2,\nu'_2)$. The projected coordinates are

```math
(u_1,v_1)=b_1R'_1\bigl(\cos(\nu'_1+\omega),\cos i\sin(\nu'_1+\omega)\bigr),
\qquad
(u_2,v_2)=-b_2R'_2\bigl(\cos(\nu'_2+\omega),\cos i\sin(\nu'_2+\omega)\bigr).
```

The disk separation is the Euclidean distance between these coordinates. The secondary is in front near primary conjunction ($w_1<0$). For the first four photometry groups omit the light travel correction by setting $\ell=0$ and $T_c=T_0$, as an explicit approximation of this conditional model.

The primary orbit-only radial velocity before its unknown zero point is

```math
V(t)=\frac{ab_1\sin i}{\eta P\sqrt{1-e^2}}
\left[\cos(\nu'_1+\omega)+e\cos\omega\right].
```

Consequently its actual semiamplitude is $(\kappa/\eta)K_1$, rather than exactly $K_1$. Keep the two supplied decimal constants unchanged.

For a disk of radius $r$, quadratic limb darkening has brightness proportional to

```math
U(\mu)=1-u(1-\mu)-v(1-\mu)^2,\qquad
\mu=\sqrt{\max(0,1-(x^2+y^2)/r^2)}.
```

Here $x,y$ are coordinates about that disk centre and $u,v$ are the specified coefficients. $S_J$ is a ratio of mean disk surface brightnesses, not central brightnesses. The uneclipsed infrared flux ratio is $S_J(r_2/r_1)^2$. Optical photometry treats the companion as dark. Subtract the fraction of the background disk flux blocked by its foreground disk, then normalize by the combined uneclipsed flux. Convert relative flux $F$ to differential magnitude $m=-2.5\log_{10}F$.

The observation operator includes the published sparse quadrature, which is an approximation to continuum limb darkening. Its definition is provided here to make the discretization reproducible. A unit disk uses eight Gauss–Legendre ordinates in $y$. For increasing positive absolute ordinate, the corresponding inner $x$ orders are 8, 7, 5, 2; the negative ordinates use the same respective orders. Scale the inner nodes to $[-\sqrt{1-y^2},\sqrt{1-y^2}]$. Normalize the brightness integral by its quadrature area. For a smaller foreground disk entirely inside the background disk, integrate background brightness over that smaller disk using these same nodes and the actual projected coordinate offsets. Do not rotate this finite disk grid to the separation direction.

For partial overlaps, rotate coordinates so the foreground centre is at $(d,0)$, where $d$ is the centre separation. Let $R_b,R_f$ be background and foreground radii, and let $c_h=(R_b^2-R_f^2+d^2)/(2d)$ be the common chord abscissa. The overlap consists of the background cap on $[c_h,R_b]$ and foreground cap on $[d-R_f,c_h]$. Each cap has vertical limits given by its own circle. If the overlap exceeds half the background area, integrate the visible region instead: the two strips between the circle boundaries on $[d-R_f,c_h]$, plus the background cap on $[-R_b,d-R_f]$. Normalize the integrated brightness by the integrated area, then multiply by the exact geometric area of the selected region. For a region with abscissa range $[a_x,b_x]$, use outer order $\operatorname{clip}(\operatorname{nint}(4|b_x-a_x|/R_b),4,8)$. With that order $n$, half-range $h_x=(b_x-a_x)/2$, and vertical limits $a_y,b_y$ at each abscissa, use inner order $\operatorname{clip}(\operatorname{nint}(n|b_y-a_y|/(2h_x)),4,8)$. Positive half-integer orders round upward. Normalize by the full background disk flux computed with the disk rule above. Exact circle overlap is

```math
A=R_b^2\arccos\frac{d^2+R_b^2-R_f^2}{2dR_b}
+R_f^2\arccos\frac{d^2+R_f^2-R_b^2}{2dR_f}
-\frac12\sqrt{(-d+R_b+R_f)(d+R_b-R_f)(d-R_b+R_f)(d+R_b+R_f)}.
```

Handle disjoint, fully covered, and fully contained disks before this expression. Uniform brightness reduces to the exact blocked area fraction. These rules specify forward integration, rather than a required inverse algorithm.

Score all 15488 observations. Each group has its own unknown additive magnitude or velocity zero point, eliminated by weighted profiling. With observation $y_j$, model $m_j$, and uncertainty $\sigma_j$, define

```math
w_j=\sigma_j^{-2},\qquad z_g=\frac{\sum_{j\in g}w_j(y_j-m_j)}{\sum_{j\in g}w_j},\qquad
Q=\sum_{g=1}^{6}\sum_{j\in g}w_j(y_j-m_j-z_g)^2.
```

Passing requires the physical domain above and $Q\le15900.910369432995$. Six group contributions are diagnostic; improvements in one group may compensate for another. This is a conditional fit-quality budget, not a statement of 95% parameter coverage. Fixed mass ratio, limb darkening, noise model, and spherical shapes are analysis assumptions. There is no independent true radius or inclination measurement used to require exact parameter identity.

## Data Description

The public arrays preserve the original decimal observation values as float64; no fitted stellar parameters are provided.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| times | (1, 15488) | float64 | Heliocentric Julian date, days; concatenated in the six-group order below. |
| measurements | (1, 15488) | float64 | Differential magnitude for the first five groups, primary radial velocity in km/s for the last group. |
| reported_uncertainty | (1, 15488) | float64 | Reported standard error, magnitude or km/s matching the corresponding measurement. |
| group_bounds | (1, 6, 2) | int64 | Unitless start-inclusive, end-exclusive row indices for WASP, NITES, OED, BYU, KPNO, RV. |
| response_flags | (1, 15488) | int8 | Unitless model convention: 1 directly evaluated, 0 flux interpolated between direct rows within its group, -1 optical magnitude fixed to zero outside the specified phase window. |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Counts | 7968, 3490, 239, 2199, 1564, 28 | observations | WASP, NITES, OED, BYU, KPNO, RV order. |
| P | 14.2769001 | day | Fixed orbital period. |
| T0 | 2456023.26988 | HJD day | Fixed primary conjunction epoch. |
| q | 0.1968 | dimensionless | Fixed companion/primary mass ratio; conditional analysis assumption. |
| kappa | 0.019771142 | solar radius / (km/s day) | Historical coefficient mapping the unknown velocity scale to a1. |
| eta | 0.019765685498705825 | solar radius / (km/s day) | Historical orbital velocity conversion; kappa/eta = 1.0002760592995639. |
| Solar radius | 695700000 | m | Radius used for light travel time. |
| Speed of light | 299792458 | m/s | Light travel time constant. |
| Day | 86400 | s | Time conversion. |
| White primary LD | 0.3532, 0.2606 | dimensionless | Quadratic coefficients for WASP and NITES. |
| I primary LD | 0.2552, 0.2598 | dimensionless | Quadratic coefficients for OED and BYU. |
| J primary LD | 0, 0 | dimensionless | Uniform primary in KPNO infrared photometry. |
| J secondary LD | 0.1076, 0.2857 | dimensionless | Quadratic secondary coefficients in KPNO photometry. |
| Disk integration order | 8 | dimensionless | Sparse area-corrected Gaussian disk rule defined above. |
| WASP phase window | abs(((1.5 + ((t-T0)/P mod 1)) mod 1) - 0.5) < 0.02 | orbital cycle | Only these rows have nonzero optical model; all other WASP observations still contribute to Q and its zero point. |
| NITES and KPNO flags | Supplied in response_flags | dimensionless | Linear interpolation of flux before conversion to magnitude; no finite exposure duration is assumed. |
| WASP error multiplier | 0.84 | dimensionless | Conditional uncertainty scaling. |
| RV jitter | 0.02 | km/s | Added in quadrature to reported velocity uncertainty. |
| Numerical table convention | IEEE float32 columns; float64 times and model arithmetic | dimensionless | For Q, round observation values and reported errors to float32. Apply the WASP multiplier and RV jitter in float32, and compute inverse squared errors in float32. This explicitly retains the selected data analysis convention; raw arrays retain full supplied decimal precision. |
| RV zero point | One shared profiled offset for 28 velocities | km/s | No additional instrument offset is fitted. |
| Other effects | Spheres; no spots, irradiation, third light, or flux weighted velocity | dimensionless | Conditional model restrictions, not independently verified absence of these effects. |

The row order within each group is preserved. The interpolation flags are part of the known observation operator, rather than measured exposure integrations. HJD values should remain float64. The infrared light travel correction uses the full relative semimajor axis; the barycentric coordinates already include the respective mass fractions.

## Method Hints

Joint nonlinear likelihood inference can combine eclipse shape with orbital velocities, eliminating additive zero points analytically. Ensemble sampling can describe correlations among radius, impact geometry, and eccentricity. A point estimate should be evaluated by the full observation likelihood and physical constraints; different acceptable solutions need not match one chosen posterior summary.

## References

- Maxted, *ellc: A fast, flexible light curve model for detached eclipsing binary stars and transiting exoplanets*, Astronomy & Astrophysics 591, A111 (2016). [Paper](https://doi.org/10.1051/0004-6361/201628579).
- Gómez Maqueo Chew et al., *The EBLM project. II. A very hot, low-mass M dwarf in an eccentric and long-period, eclipsing binary system from the SuperWASP Survey*, Astronomy & Astrophysics 572, A50 (2014). [Observational study](https://doi.org/10.1051/0004-6361/201424265).
