# Implemented mathematics and experimental scope

The binding is five independent real natural specimens measured by paired thermal Coe experiments. Unknown B is the original cooling-field strength, in microtesla. No physical ground truth is declared. The original author's full processed fit supplies the numerical method reference. Historical expected fields are separately labeled context and are never used by the solver, public calibration or Judge.

## Discretization and preprocessing

All 181 native readings across five specimens are retained, including check rows. Input shape (1,5,46) uses explicit valid counts. Decimal treatment suffixes were converted privately to integer codes; native temperatures below 50 °C map to the original unheated point at 0 °C. Amplitudes preserve the official TDT import's decimal serialization, then divide by each first measured amplitude. This is observation-derived normalization, not hidden-target calibration. Original import precision differs from the extra digits of the LV6C3A text file; validation records this discrepancy explicitly.

For magnitude m, declination d and inclination i in radians, the Cartesian vector is m(cos d cos i, sin d cos i, sin i), with north/east/down axes. At each paired temperature, acquired partial TRM is the in-field vector minus its zero-field vector. The independent initial observation defines M0. Divide acquired-vector and zero-field-vector magnitudes by M0 to obtain dimensionless x and y. Initial x=0. All five bound protocols have one zero and one in-field partner per noninitial temperature; a missing or duplicated partner raises an error.

## Forward model and inverse

The linear thermal-reciprocity model is y=a+b x+epsilon, b=-B/Blab. The selected implementation is the published SPD signed standard-major-axis regression, not ordinary least squares or an orthogonal-distance substitute. For n points in the fixed inclusive interval, define dx=x-mean(x), dy=y-mean(y), Sxx=sum(dx²), Syy=sum(dy²), Sxy=sum(dx dy).

```math
b=\mathrm{sign}(S_{xy})\frac{\mathrm{std}(y;\mathrm{ddof}=1)}{\mathrm{std}(x;\mathrm{ddof}=1)},\quad
a=\bar y-b\bar x,\quad B=|b|B_{\mathrm{lab}},\quad
\sigma_b=\sqrt{\frac{2S_{yy}-2bS_{xy}}{(n-2)S_{xx}}},\quad
\beta=|\sigma_b/b|,\quad\sigma_B=B_{\mathrm{lab}}\sigma_b.
```

All quantities are float64. B and sigmaB use microtesla; a,b,sigmaB/Blab are dimensionless. Preserve the upstream zero-slope and zero-uncertainty guards of 1e-10. At least three paired points are required. There is no regularizer, training stage, initialization, iteration, step size, line search or stopping rule: this inverse evaluates a closed-form formula. The signed regression can diagnose a wrong positive slope; the published field report uses absolute slope.

## Directional and acquisition diagnostics

For selected zero-field vectors z, anchored PCA diagonalizes sum(z zᵀ); free PCA subtracts their mean first. Sort eigenvalues decreasingly, choose the leading unit eigenvector with sign matching the initial magnetization, and compute MAD=atan(sqrt((lambda2+lambda3)/lambda1)) in degrees. A PSD roundoff guard clamps only a negative roundoff numerator to zero for an exactly rank-one independent diagnostic; all measured-case eigenvalues and parity remain unchanged. This edge handling is explicitly distinct from the upstream unguarded square root.

For full zero-field vector series, VDS=sum(norm(z[j+1]-z[j]))+norm(z[last]); FRAC is the sum of successive vector changes inside the selected interval divided by VDS. It is not the endpoint norm divided by interval path length.

Repeat pTRM coordinates are norm(check vector - immediately previous measured vector), paired with the zero-field magnitude at the checked temperature. Tail-check coordinates use acquired TRM at the matching temperature and the directly measured tail magnitude. Normalize by M0. A check enters SCAT only if both its check and preceding-treatment temperatures lie in the chosen interval.

SCAT uses betaThreshold=.1. Let s1=b+2|b|betaThreshold, s2=b-2|b|betaThreshold; c1=mean(y)-s1 mean(x), c2=mean(y)-s2 mean(x), u1=-c1/s1 and u2=-c2/s2. The lower boundary is c1-(c1/u2)x and upper c2-(c2/u1)x. Every selected pair and eligible check must have nonnegative x,y, x<=u1, y<=c2, and lie between those lines. This implements the original acquisition checks without silently discarding failures.

## Scope, parameters and evaluation

The original full SPD run also calculates additional descriptive statistics. This task selects its SMA inverse and PCA/FRAC/SCAT diagnostics explicitly; curvature fitting is not part of the inverse or a reported fresh result. No anisotropy, nonlinear-TRM correction or microwave experiment is selected. The original RS25b corrected MATLAB example remains a private cross-oracle diagnostic; it is not claimed identical to uncorrected GitHub inference.

All five fitting intervals are acquisition inputs. Original SCAT=true and FRAC>=.5 plus natural thermal material and no required corrections fixed specimen selection before extraction. The notebook recomputes the published SMA path, an endpoint-trim sensitivity study, OLS as a labeled statistical diagnostic, and scalar-magnitude subtraction as a labeled physically incomplete diagnostic. Each alternative is actually calculated from observations. Historical field discrepancy demonstrates remaining model bias.

The evaluator requires float64(1,5) and maximum absolute field discrepancy <=1e-6 microtesla against the independently reproduced method result. No scale, sign, crop, mask or alignment adjustment applies. This is a numerical tolerance; physical uncertainty is considerably larger. Zero self-comparison is not converted into an impossible zero boundary.
