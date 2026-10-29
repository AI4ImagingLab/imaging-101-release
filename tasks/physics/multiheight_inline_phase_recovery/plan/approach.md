# Implemented method

The unknown u is a dimensionless complex scalar field on a 512 by 512 object-plane grid. Pixel spacing d=1.12 micrometres, wavelength lambda=0.532 micrometres, and object-to-detector distances z=(300,350,400,450,500,550,600,650) micrometres are fixed. Wave number k=2*pi/lambda has units radians per micrometre. The eight uint16 intensity images are existing author simulations. No new observations or target-derived calibration are generated.

## Observation preprocessing

`amplitude_stack` preserves the original ImageToArray/Normalize/Map/MakeComplex pipeline: convert camera counts D to float32, divide by 65535, subtract each plane's spatial minimum m, divide by 1-m, take the square root, and cast to complex64 with zero imaginary part. The amplitude observations are a[j]. The retained per-plane normalization changes ideal propagation consistency, so a nonzero observation residual is expected.

## Discrete acquisition operator

Only positive even two-dimensional shapes are supported. For each axis of length n, define angular frequencies by float64 linspace from -n/(2*d*n) through (n/2-1)/(d*n), multiply by 2*pi, apply fftshift, then construct an ij-indexed meshgrid and cast it to float32. Squared transverse frequency Q is computed from that float32 grid. The bound 512-square acquisition has only propagating frequencies. The original odd-size fftshift convention is not validated as a physical frequency grid and is explicitly rejected.

```math
Q=q_x^2+q_y^2,\qquad \Delta=k^2-Q,
\qquad H_z=
\begin{cases}
\exp(i z\sqrt{\Delta}) & \Delta\geq 0,\\
\exp(-|z|\sqrt{-\Delta}) & \Delta<0.
\end{cases}
```

Here qx and qy are angular spatial frequencies, Delta is squared longitudinal wave number, z is propagation distance, and Hz is the transfer function. Local propagation evaluates ifft2(H*fft2(u)), with NumPy's default unscaled forward FFT and normalized inverse FFT. The spatial domain is periodic; there is no padding or bandlimit mask. For this acquisition the operator is unitary to floating-point precision. Constant-field, energy, inverse, and adjoint tests check signs and normalization independently.

## Inverse updates and stopping

Initialize h=a[0], the first detector amplitude with zero phase. Store signed backpropagation distances v=-z as NumPy float64 scalars. One forward sweep visits j=0,...,6, propagates by v[j]-v[j+1], and replaces amplitude with a[j+1]. One reverse sweep visits j=7,...,1, propagates by v[j]-v[j-1], and replaces amplitude with a[j-1].

```math
r=\mathcal{P}_{v_j-v_{j+1}}h,
\qquad h\leftarrow a_{j+1}\exp(i\arg r)
```

The reverse sweep has the same expression with j+1 replaced by j-1. Perform exactly 100 complete forward/reverse sweeps, then return object-plane field P(v[0])h. The selected modern Fringe implementation is an unweighted multi-distance amplitude-projection extension of Gerchberg-Saxton, consistent with the cited Pedrini multi-plane method. There is no spatial regularizer, support constraint, line search, learned prior, or positivity operation on the complex field. The projection determines the update directly; no step-size parameter or adaptive stopping is used. No claim of global minimization is made.

The monitored amplitude mismatch and its normalized residual are

```math
E(u)=\sum_j\left\lVert |\mathcal{P}_{z_j}u|-a_j\right\rVert_2^2,
\qquad R(u)=\frac{\sqrt{E(u)}}{\sqrt{\sum_j\lVert a_j\rVert_2^2}}.
```

There is no 1/M or 1/2 normalization in E. Record R every five sweeps without changing the updates. A fresh single-plane backpropagation with zero sweeps and a fresh two-plane, 30-sweep reconstruction are notebook comparisons; the full benchmark remains eight planes and 100 sweeps.

## Numerical conventions

Preserve NumPy float64 scalar distances in full reconstruction, predictions, monitoring, and Judge. They promote the source float32 frequency root to complex128 transfer kernels. Python float distances instead produced complex64 kernels in NumPy 2.4.6 and a maximum field difference approximately 0.154 after 100 sweeps. That mismatch was investigated and corrected without relaxing tolerances. Frozen literal integer-distance stage cases deliberately preserve their captured complex64 argument path; they are not the full reconstruction configuration.

The diagnostic prediction audit found that using Python float only in the prediction helper changed the residual from 0.06764382199912759 to the precision-consistent 0.06764432302086566, with maximum amplitude-map difference 0.0002331873602132717. The final prediction and independent Judge preserve float64 distances. The reconstructed field and upstream fixtures are unchanged.

## Evaluation conventions

The hidden reference is the independently executed modern source method, not actual truth or a historical paper result. Original reruns are bitwise identical. Align only a constant unit phase exp(-i*arg(vdot(reference,estimate))). Complex NCC is abs(vdot(reference,estimate))/(norm(reference)*norm(estimate)); complex NRMSE is RMS magnitude of the aligned field difference divided by max(abs(reference))-min(abs(reference)). No gain, translation, phase ramp, or crop is fitted.

Require NCC >= 0.999999 and NRMSE <= 0.0001. These nonzero numerical tolerances allow backend roundoff despite zero source self-error. Also require R <= 0.06832026206177774, fixed before local acceptance as 1.01 times the original diagnostic residual 0.06764382382354232. The parent precision audit did not change this boundary. These checks establish reproducibility and observation agreement, not physical truth accuracy.
