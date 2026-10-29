# Implemented mathematics

This task selects the complete current author closed-form CPU pipeline, not the optional iterative Poisson/TV algorithm. Real data are all512x32x32 photon-count values. Geometry recovery is conditional on the author effective discretization; no actual voxel truth, calibrated absolute albedo or absolute clock is available.

## Observation preprocessing

Let m[t,i,j] be stored photon counts, with native temporal index t and lateral indices i,j. Use original indices0...416; linearly interpolate these417 inclusive samples onto128 inclusive locations spanning the same interval, in float64. Divide by the global maximum then cast float32. Zero extension puts this array in the leading128x32x32 block of256x64x64. This preserves all scan positions. The remaining95 stored native time bins are outside the selected original reconstruction window; they are not removed from public observations.

## Diffusive acquisition model

The physical approximation is y=M(phi*phi*A rho)+noise: A describes round-trip free-space transport of relative object response rho, phi is a slab flux impulse, M crops the measured window. Convolution acts in time and both lateral coordinates. A point has geometric round-trip delay2r/c0 for distance r, with illumination/detection radiometric factor proportional to r^-4 in the independent point controls. The closed-form migration uses an approximate wave-envelope inverse rather than an exact inverse of a dense albedo matrix.

Set c0=3e8m/s, n=1.12, v=c0/n, reduced scattering mu-s=262m^-1, absorption mu-a=.53m^-1, thickness d=.0254m and extrapolation ze=.0036m. D=1/[3(mu-a+mu-s)] has units m; source depth z0=1/mu-s. Kernel lateral coordinates u,w span[-.7,.7] on64 inclusive points. Model time has256 inclusive points spanning0...4/v seconds. At t=0 use the source1-second placeholder: exponential attenuation underflows to zero. For image integer q=-20...20, define:

```math
a_q=d(1-2q)-4qz_e-z_0,\qquad
b_q=d(1-2q)-(4q-2)z_e+z_0.
```

The variables a and b are image-source-to-exit signed distances. The unnormalized single-pass flux is:

```math
g(t,u,w)=(4\pi Dv)^{-3/2}t^{-5/2}
\exp\left[-\mu_a vt-\frac{u^2+w^2}{4Dvt}\right]
\sum_{q=-20}^{20}\left[a_q\exp\left(-\frac{a_q^2}{4Dvt}\right)-b_q\exp\left(-\frac{b_q^2}{4Dvt}\right)\right].
```

The source omits a common1/2 in the published flux formula; normalization removes it exactly. Divide g by its total discrete sum; roll both lateral axes by-32; form double-pass kernel h=abs(IFFT(FFT(g)^2)). Kernel construction is float64 NumPy, then h is cast float32. The absolute value is the original convention; independent physical-series negative flux and FFT roundoff are quantified separately. The exact transpose of circular convolution multiplies by conjugate H, where H=FFT(h); tests verify its dot product and constant-flux conservation.

## Author Wiener arithmetic

The original implementation stores complex coefficients as float32 pairs. Compute conjugate H and its pairwise complex product P=H*conj(H), add1/alpha to **both real and imaginary channels**, take componentwise reciprocal, then pair-multiply by conj(H). Alpha=2500 is the selected original regularization. This is not conventional complex reciprocal: mathematically the resulting response is conj(H)/(|H|^2+1/alpha)+i*alpha*conj(H). Hermitian H makes the second real-space component purely imaginary. Only the real inverse FFT is retained, so exact real arithmetic agrees with Wiener filtering; float32 cancellation is a known risk. Preserve the actual arithmetic rather than silently claiming identical intermediate complex coefficients. Alternate complete source FFT-backend runs determine prelocal numerical bounds. All FFTs use unnormalized forward transform and1/N inverse normalization.

## Wave migration

Start with sqrt(max(deconvolved,0)); the current source omits the t multiplier written in SI pseudocode. This discrepancy is disclosed, and independent physical point controls establish nominal geometric resolution. Let M=128,N=32, normalized grids z=-M...M-1 divided by M and x,y=-N...N-1 divided by N. Set width=.7m and padded path range L=4m. Define:

```math
\gamma=\frac{NL}{4M\,\mathrm{width}},\quad
q=\sqrt{\gamma^2(x^2+y^2)+z^2},\quad
p=M(q+1).
```

Here q is normalized positive temporal-frequency coordinate and p its source-array index. FFT-shift the transformed square-root envelope. Interpolate complex values at p on each existing lateral-frequency index, using floor/ceil linear weights; p outside[0,255] contributes zero. Both lateral coordinates are integer, so no lateral interpolation is needed. Multiply by abs(z)/max(q,1e-8), discard z<=0, inverse-shift and inverse-transform. Square real and imaginary parts separately and add. Crop leading128x32x32; set final15depth slices to zero as in the author output. This source boundary suppression is explicit in the public contract. No iterative initialization, optimizer, stopping criterion or training is used: the inverse is closed form.

Migration-only and signed deconvolution-only are computed independently on the same fresh observations, preserving all original comparisons. Notebook additionally computes full SNR250/2500/25000 and mu-s0.7x/1x/1.3x ablations; these are sensitivity diagnostics, not alternative benchmark definitions.

## Units, discrepancies and acceptance

Native16ps is a source-comment nominal value. It implies resampled interval52.4094ps, while the effective kernel uses58.5621ps. The inclusive64-point kernel lateral spacing also differs from the32-point scan spacing. The author plotted output depth0...1m is retained as **nominal coordinate**, not validated absolute range. Independent analytic diffusion and geometric points at.3,.5,.75m meet paper-derived9cm axial/15cm lateral resolution; they do not prove absolute calibration of the real data or full-volume albedo accuracy. Hidden reference is the complete original method response. Prefrozen gates are cosine NCC>=.99999, dynamic-range NRMSE<=.0002 and max/range<=.0025, plus zero negative and boundary counts. These approximately triple independently measured alternate-FFT envelopes; they were not fitted to cleaned output. No masks, scale fitting, translations or implicit alignment are applied. Metric quantities use float64 and explicitly reject degenerate references and nonfinite scoring.
