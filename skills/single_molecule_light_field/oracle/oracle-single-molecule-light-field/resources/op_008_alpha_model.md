# Phase-averaged sphere alpha (INTEGRATE_SPHERE), m=10

na=1.27, n=1.33, na_n=0.95489. Aperture width uv_scaling = mla_to_uv_scale = 0.645219, ds2 = 0.322610.

For a pupil point (u,v):
  grid g = linspace(-ds2, ds2, m+1)
  u' = u + g (broadcast over one axis), v' = v + g (other axis)
  rho2 = u'^2 + v'^2
  radicand = 1 - rho2*na_n^2 ; set to NaN where < 0
  phi = -na_n / sqrt(radicand)
  alpha_U = nanmean(u' * phi) ; alpha_V = nanmean(v' * phi)

Reference alpha for the 7 views (u,v -> alpha_U, alpha_V):
  ( 0.0000,  0.0000) -> ( 0.0000,  0.0000)
  ( 0.6452,  0.0090) -> ( 1.0594,  0.0162)
  (-0.6452, -0.0090) -> (-1.0594, -0.0162)
  ( 0.3304, -0.5542) -> ( 0.5110, -0.9050)
  (-0.3304,  0.5542) -> (-0.5110,  0.9050)
  ( 0.3148,  0.5632) -> ( ~0.49,   ~0.90 )
  (-0.3148, -0.5632) -> ( ~-0.49,  ~-0.90)
The batched (vectorised) and per-point loop implementations agree to ~1e-16.
