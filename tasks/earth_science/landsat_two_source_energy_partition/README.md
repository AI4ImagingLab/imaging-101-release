# Landsat Two-Source Energy Partition

> Reconstruct soil and canopy temperatures and their sensible and latent heat exchange from mixed satellite thermal radiance and meteorological forcing.
> Domain: earth science | Keywords: thermal remote sensing, energy balance, evapotranspiration | Difficulty: Hard

## Background

A satellite thermal pixel combines radiation emitted by vegetation and exposed ground. Different soil and leaf temperatures can produce the same mixed brightness temperature, so component temperatures are not directly measured. Optical vegetation information helps determine which component is visible, while weather observations constrain atmospheric heat transfer.

Separating these components matters for irrigation, drought assessment and land-atmosphere exchange. Sensible heat warms the air, latent heat carries energy consumed by evaporation and transpiration, and part of the available energy enters the ground. Coupled energy budgets constrain the inverse beyond a mixed thermal observation.

## Problem Description

**Scientific contract status: not certified for benchmark release.** The current evaluation requires a finite approximate PT/stability protocol whose result is not a uniquely defined physical closure solution. The physical model, observations and output layout below remain documented; historical method-comparison criteria are not a certified public challenge.

Reconstruct the complete finite acquisition footprint of an actual Landsat9 scene with ERA5 forcing. The directional thermal measurement is
$$T_r^4=f_v T_c^4+(1-f_v)T_s^4.$$
Here all temperatures are in kelvin: $T_r$ is the mixed radiometric temperature, $T_c$ canopy temperature, $T_s$ soil temperature, and $f_v$ dimensionless visible vegetation fraction derived from the optical observation and geometry. Component energy budgets and the resistance network are
$$S_c+L_c=H_c+LE_c,\qquad S_s+L_s=H_s+LE_s+G,$$
$$H_c=\rho c_p(T_c-T_a^\ast)/R_x,\quad H_s=\rho c_p(T_s-T_a^\ast)/R_s,\quad H_c+H_s=\rho c_p(T_a^\ast-T_a)/R_a.$$
Subscripts c/s mean canopy/soil. $S,L$ denote net shortwave/longwave radiation; $H,LE$ sensible/latent heat; $G$ ground heat, all in W m^-2. Positive radiation enters the surface, positive H/LE leaves it and positive G enters the ground. $T_a$ is above-canopy air temperature, $T_a^\ast$ canopy-interface air temperature, $\rho$ air density in kg m^-3, $c_p$ specific heat in J kg^-1 K^-1, and resistances $R_x,R_s,R_a$ are in s m^-1. Radiation, atmospheric stability and vegetation partition couple the unknowns. The single thermal observation cannot uniquely determine them without physical assumptions.

### Physical closure laws

The target is the canonical finite two-source Priestley-Taylor estimator (PT), with a one-source energy balance (OSEB) for bare pixels. It is not an arbitrary exact solution of the budgets above. The following fixed physical laws define its nominal model; the finite temperature approximation and stopping conventions are specified in Method Hints.

**Thermodynamics.** Write p for the observed pressure divided by100, in mb, and $T_d$ for dewpoint in K. Define vapour pressure $e=6.112\exp[17.67(T_d-273.15)/(T_d-29.65)]$ mb, specific humidity $q=.622e/[p+(.622-1)e]$, and
$$c_p=(1-q)1003.5+1865q,\qquad
\rho=\frac{100p}{287.04T_a}\left[1-(1-.622)e/p\right].$$
Here cp is in J kg^-1 K^-1 and rho in kg m^-3. With $t=T_a-273.15$ in Celsius, latent heat $\lambda_v=10^6(2.501-.002361t)$ J kg^-1, saturation-pressure slope $\Delta=10(4098)(.6108)\exp[17.27t/(t+237.3)]/(t+237.3)^2$ mb K^-1, and psychrometric coefficient $\gamma=c_pp/(.622\lambda_v)$ mb K^-1. The canopy-transpiration constraint and initial ground-heat relation are
$$H_c=R_{n,c}\left[1-\alpha f_g\frac{\Delta}{\Delta+\gamma}\right],
\qquad R_{n,c}=S_c+L_c,\quad R_{n,s}=S_s+L_s,\qquad G=.35R_{n,s},$$
where alpha is the adjustable Priestley-Taylor coefficient and fg the stated green fraction. Latent components are residuals of their separate energy budgets. The finite daytime dry limit is described below.

**Vegetation visibility.** Let F denote the NDVI-derived leaf area index in the metadata and fc the nominal canopy cover used for clumping (fc=1 here). For zenith angle theta and ellipsoidal leaf parameter x, beam extinction is
$$K(\theta)=\frac{\sqrt{x^2+\tan^2\theta}}{x+1.774(x+1.182)^{-.733}}.$$
Angles in this expression and the following exponent are in radians. Local leaf area is $F_\ell=F/f_c$. Nadir clumping $\Omega_0=-\log[f_c\exp(-K(0)F_\ell)+1-f_c]/[F_\ell K(0)]$ and angular clumping is
$$\Omega(\theta)=\frac{\Omega_0}{\Omega_0+(1-\Omega_0)\exp[-2.2\theta^{\,3.8-.46/w_c}]},
\qquad f_v=1-\exp[-K(\theta_v)\Omega(\theta_v)F_\ell].$$
Here wc is canopy width/height and theta-v the view zenith. This task has fc=wc=x=1 and nadir view. F=0 is handled as bare soil, without an undefined canopy temperature.

**Incoming shortwave partition.** The nominal solar zenith derived from the listed acquisition geometry is29.07255314015575degrees, uniform over the footprint. With its cosine mu, optical airmass m=1/mu, extraterrestrial visible irradiance V=1320(1-.5455), NIR irradiance N=1320*.5455, and p0=1013.25mb, define horizontal potential components
$$B_v=V e^{-.185(p/p_0)m}\mu,\quad D_v=.4(V\mu-B_v),\quad
W=1320\,10^{-1.195+.4459\log_{10}m-.0345(\log_{10}m)^2},$$
$$B_n=\max[0,(Ne^{-.06(p/p_0)m}-W)\mu],\qquad
D_n=\max[0,.6((N-W)\mu-B_n)].$$
B/D denote beam/diffuse and v/n visible/NIR; W is normal-plane water absorption in W m^-2. All B/D components are horizontal W m^-2. These are the restored unit-consistent atmospheric equations used by this task.

Set $P_v=B_v+D_v$, $P_n=B_n+D_n$, visible fraction $f_{\mathrm{vis}}=P_v/(P_v+P_n)$, NIR fraction $f_{\mathrm{nir}}=1-f_{\mathrm{vis}}$, and clear index $\Phi=\min(1,S_\downarrow/(P_v+P_n))$, where Sdown is the observed downward shortwave. Beam fractions within each spectral band are
$$b_v=\operatorname{clip}\left[\frac{B_v}{P_v}\left(1-\left(\frac{.9-\min(\Phi,.9)}{.7}\right)^{.6667}\right),0,1\right],$$
$$b_n=\operatorname{clip}\left[\frac{B_n}{P_n}\left(1-\left(\frac{.88-\min(\Phi,.88)}{.68}\right)^{.6667}\right),0,1\right].$$
Clip limits a scalar to the stated interval. The total diffuse fraction is $\xi=(1-b_v)f_{\mathrm{vis}}+(1-b_n)f_{\mathrm{nir}}$. Actual incident beam and diffuse irradiances are $S_b=S_\downarrow(1-\xi)$ and $S_d=S_\downarrow\xi$.

**Canopy spectral transfer.** For each visible/NIR band, use the stated leaf reflectance r-leaf, leaf transmittance t-leaf and soil reflectance r-soil. Define absorptivity $a=1-r_{\mathrm{leaf}}-t_{\mathrm{leaf}}$, deep-canopy reflectance $r_\infty=(1-\sqrt a)/(1+\sqrt a)$, and $b(k)=2kr_\infty/(k+1)$. For extinction k and effective leaf area A, let $E=\exp(-\sqrt a\,kA)$ and
$$\tau(k,A)=\frac{(b^2-1)E}{br_{\mathrm{soil}}-1+b(b-r_{\mathrm{soil}})E^2},\quad
Q=\frac{(b-r_{\mathrm{soil}})E^2}{br_{\mathrm{soil}}-1},\quad
\mathcal A(k,A)=\frac{b+Q}{1+bQ}.$$
Here tau is transmittance, cal-A albedo and b=b(k). Beam transfer uses $k=K(\theta_s)$ and $A=F\Omega(\theta_s)$, with solar zenith theta-s. For diffuse transfer, define
$$t_0=2\sum_{\theta=0,5,\ldots,85\ \mathrm{deg}}
e^{-K(\theta)F}\cos\theta\sin\theta\,\delta\theta,\quad
\delta\theta=5\pi/180,\qquad k_d=-\log(t_0)/F,$$
using radians inside the functions; diffuse transfer uses k=kd,A=F. With tau-b/tau-d and albedo-b/albedo-d from these expressions,
$$S_c=\sum_{j\in\{\mathrm{vis},\mathrm{nir}\}} f_j
[(1-\tau_{b,j})(1-\mathcal A_{b,j})S_b+(1-\tau_{d,j})(1-\mathcal A_{d,j})S_d],$$
$$S_s=\sum_j f_j(1-r_{\mathrm{soil},j})
[\tau_{b,j}S_b+\tau_{d,j}S_d].$$
At bare pixels use $S_c=0$ and $S_s=S_\downarrow[1-f_{\mathrm{vis}}r_{\mathrm{soil,vis}}-f_{\mathrm{nir}}r_{\mathrm{soil,nir}}]$.

**Thermal radiation.** Here epsilon-c and epsilon-s denote the listed canopy and soil emissivities. Compute diffuse thermal albedo $\mathcal A_L$ and transmission $\tau_L$ with the same transfer equations, replacing leaf reflectance by $1-\epsilon_c$, leaf transmittance by0 and soil reflectance by $1-\epsilon_s$. With emitted irradiances $E_c=\epsilon_c\sigma T_c^4$, $E_s=\epsilon_s\sigma T_s^4$, observed downward longwave $L_\downarrow$ and stated Stefan-Boltzmann constant sigma, net longwave is
$$L_s=\epsilon_s\tau_LL_\downarrow+\epsilon_s(1-\tau_L)E_c-E_s,$$
$$L_c=(1-\mathcal A_L)(1-\tau_L)(L_\downarrow+E_s)-2(1-\tau_L)E_c.$$
At bare pixels $L_c=0,L_s=\epsilon_s(L_\downarrow-\sigma T_r^4)$.

**Wind and thermal resistances.** Let h be canopy height, d displacement, z0 momentum roughness, z0H heat roughness, zu/zT observation heights and ell leaf width, all in m. The observed wind speed u is in m s^-1. Von Karman constant kappa=.41. Friction velocity u-star and aerodynamic resistance obey
$$u_\ast=\frac{\kappa u}{\log((z_u-d)/z_0)-\Psi_M((z_u-d)/L)+\Psi_M(z_0/L)},$$
$$R_a=\frac{\log((z_T-d)/z_{0H})-\Psi_H((z_T-d)/L)+\Psi_H(z_{0H}/L)}{\kappa u_\ast}.$$
Canopy-top wind is $U_c=u_\ast[\log((h-d)/z_0)-\Psi_M((h-d)/L)+\Psi_M(z_0/L)]/\kappa$. Goudriaan attenuation is $U(z)=U_c\exp[-a_G(1-z/h)]$, with $a_G=.28F^{2/3}h^{1/3}\ell^{-1/3}$. The selected component resistances are
$$R_x=\frac{90}{F}\sqrt{\frac{\ell}{U(d+z_0)}},\qquad
R_s=\left[.0038\max(T_s-T_a^\ast,0)^{1/3}+.012U(.01)\right]^{-1}.$$
Use the declared nominal geometry. The canopy-interface temperature is the resistance-weighted mean
$$T_a^\ast=\frac{T_a/R_a+T_s/R_s+T_c/R_x}{1/R_a+1/R_s+1/R_x}.$$

**Atmospheric stability.** The Monin-Obukhov length L is
$$L=-\frac{\rho c_pT_au_\ast^3}{.41(9.8)H_v},\qquad
H_v=H_c+H_s+.61T_ac_p(LE_c+LE_s)/\lambda_v.$$
L is in m; zero virtual sensible flux Hv means neutral L=infinity. For dimensionless zeta=z/L, the Brutsaert functions are zero at neutral/nonfinite zeta. For finite zeta>=0, both equal $-6.1\log[\zeta+(1+\zeta^{2.5})^{1/2.5}]$. For zeta<0 let y=-zeta. Heat correction is $\Psi_H=(.943/.78)\log[(.33+y^{.78})/.33]$. For momentum let a=.33,b=.41,$x=(y/a)^{.333333}$ and $Y=\min(y,b^{-3})$; the finite convention computes x before capping y. Then
$$\Psi_M=\log(a+Y)-3bY^{.333333}
+\frac{ba^{.333333}}2\log\frac{(1+x)^2}{1-x+x^2}
+\sqrt3\,ba^{.333333}\arctan\frac{2x-1}{\sqrt3}+\Psi_0,$$
where $\Psi_0=-\log a+\sqrt3\,ba^{.333333}\pi/6$. The rounded exponent and this cap order belong to the specified finite model.

Save output/reconstruction.npz with these finite real arrays. Pixels follow ascending public pixel_index. The canopy-only axis contains precisely the public pixels with positive derived leaf area index; a pixel modeled as bare has no canopy temperature.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 110935, 4) | float64 | [LEc,Hc,LEs,Hs], W m^-2; batch,pixel,component |
| soil_temperature | (1, 110935) | float64 | Positive soil temperature, K |
| canopy_temperature | (1, 110929) | float64 | Positive canopy temperature, K; positive-LAI pixels in pixel order |
| soil_heat_flux | (1, 110935) | float64 | Ground heat G, W m^-2, positive downward |
| radiation | (1, 110935, 4) | float64 | [Sc,Ss,Lc,Ls], W m^-2, net incoming |

Reconstruct all110935finite input pixels, including6pixels modeled as bare because their derived leaf area index is zero. Their canopy radiation/fluxes are exactly zero under this assumption; the optical proxy is not independently measured land-cover truth. No fitted-pixel exclusion, rescaling, registration or post-hoc alignment is applied. Evaluation compares every declared quantity at its physical scale and checks full mixed temperature and both component energy balances. Method agreement is not independent ground-measurement accuracy.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| NDVI | (1, 110935) | float32 | Landsat normalized difference vegetation index, dimensionless |
| radiometric_temperature | (1, 110935) | float32 | Retrieved mixed surface temperature, K |
| dewpoint_temperature | (1, 110935) | float32 | ERA5 dewpoint forcing, K |
| surface_pressure | (1, 110935) | float32 | ERA5 surface pressure, Pa |
| air_temperature | (1, 110935) | float32 | ERA5 air temperature forcing, K |
| wind_speed | (1, 110935) | float32 | ERA5 wind forcing, m s^-1 |
| solar_radiation | (1, 110935) | float32 | Downward shortwave forcing, W m^-2 |
| thermal_radiation | (1, 110935) | float32 | Downward longwave forcing, W m^-2 |
| pixel_index | (1, 110935) | int64 | Zero-based row-major index in336by335image, dimensionless |

Preserve source float32 precision during preprocessing; promotion before nonlinear transforms can change iteration branches. The image has1625unavailable acquisition pixels. The footprint is selected from finite observations before fitting, never from reconstruction quality. Satellite retrievals and ERA5 forcings are derived observations, not independent flux truth.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Scene | LC09_172039_20230421 | none | Landsat9 on2023-04-21 |
| Day;local observation time | 111;10.98 | day;hour | Author observation convention |
| Nominal latitude;longitude | 30.25;38.25 | degree north;degree east | Single solar location for footprint |
| Standard meridian | 45 | degree east | Nominal local-time convention |
| Viewing zenith | 0 | degree | Nadir mixing |
| Image grid | 336rows,335columns,30spacing | pixel;pixel;m | index=row*335+column |
| Map coordinates | EPSG32637;first center422850,3351750 | m easting;m northing | x=422850+30column,y=3351750-30row |
| Wind;temperature heights | 10;2 | m | Above local ground |
| Canopy height;leaf width | 0.3;0.1 | m | Nominal assumptions, not measured site truth |
| Leaf;soil emissivity | 0.98;0.95 | none | Nominal thermal properties |
| Leaf reflectance PAR;NIR | 0.07;0.32 | none | Nominal spectral properties |
| Leaf transmittance PAR;NIR | 0.08;0.33 | none | Nominal spectral properties |
| Soil reflectance PAR;NIR | 0.15;0.25 | none | Nominal spectral properties |
| Green fraction;clumping cover | 1;1 | none | Nominal vegetation assumptions |
| Canopy width/height;ellipsoidal leaf parameter | 1;1 | none | Spherical leaf distribution |
| Roughness;displacement | crop z0=h/8,d=0.65h;bare z0=0.01,d=0 | m | Nominal landcover |
| Heat roughness | z0H=z0M | m | Nominal kB=0 |
| Leaf area index | -ln(1-clip(NDVI-0.05,0,1))/0.5 | m^2 m^-2 | Derived optical approximation |
| Pressure conversion;standard pressure | Pa/100;1013.25 | mb;mb | Atmospheric normalization |
| Solar constant;extraterrestrial NIR fraction | 1320;0.5455 | W m^-2;none | Empirical spectral partition |
| Stefan-Boltzmann constant | 5.670373e-8 | W m^-2 K^-4 | Thermal emission convention |
| Observation noise | Not supplied | none | No independent pixel error covariance |
| Spatial boundary | Independent pixel budgets | none | No lateral heat transport |
| Physical closure constants | Priestley--Taylor coefficient1.26; minimum friction/canopy/within-canopy wind0.01m/s; minimum resistance0.1s/m | dimensionless; m/s; s/m | Adopted physical closure guards, not an iteration or arithmetic protocol. |

## Method Hints

A two-source surface-energy-balance family partitions radiometric temperature and turbulent/latent fluxes between canopy and soil. Priestley--Taylor canopy closure and atmospheric stability coupling can regularize this underdetermined partition. Treat bare pixels with a one-source physical limit and retain the declared physical output masks and energy units. The current finite-protocol estimator is a maintainer comparison only; its public benchmark contract is not yet certified.

## References

- Norman,Kustas,Humes(1995), [doi:10.1016/0168-1923(95)02265-Y](https://doi.org/10.1016/0168-1923(95)02265-Y).
- Kustas,Norman(1999), [doi:10.1016/S0168-1923(99)00005-2](https://doi.org/10.1016/S0168-1923(99)00005-2).
- Peng etal.(2023), detailed two-source mathematics, [doi:10.1016/j.isprsjprs.2023.03.009](https://doi.org/10.1016/j.isprsjprs.2023.03.009), AppendicesA-C.
- Weiss,Norman(1985), [doi:10.1016/0168-1923(85)90020-6](https://doi.org/10.1016/0168-1923(85)90020-6).
