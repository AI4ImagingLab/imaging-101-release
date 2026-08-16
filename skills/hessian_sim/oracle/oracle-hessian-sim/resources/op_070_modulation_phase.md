# Modulation depth + phase

## Cutoffs
fc_ang = ceil(120*n/512)  (wide, phase)
fc_con = ceil(105*n/512)  (narrow, depth); ceil(80*n/512) if wavelength==647

## Coordinate rescale
scale = n / floor((zuobiaox[0,0] + zuobiaoy[0,0]) / 2)
zx512 = zuobiaox*scale; zy512 = zuobiaoy*scale

## Band separation here uses inv_pm (same phase matrix) but NO amplitude normalise, NO support mask

## Per order spi (0..3), spi_m = spi+1
idx_center  = ceil(spi_m/2)*3 - 1 - 1
idx_shifted = ceil(spi_m/2) + 2*floor(spi_m/2) - 1
coord_idx   = spi_m + 1 + floor((spi_m-1)/2) - 1
kytest = 2*pi*(zx512[coord_idx]-n)/(2n); kxtest = 2*pi*(zy512[coord_idx]-n)/(2n)

## Complex modulation maps (pad center/sideband to 2n first)
Ir = exp(1j*(kxtest*xx + kytest*yy))
replcHtest_* = binary(|shift_otf(H1_*_2n,kxtest,kytest,n)| > 0.9)
replch_*     = shift_otf(H_*_2n,kxtest,kytest,n) * replcHtest_*
sp_real = fftshift(ifft2(ifftshift(spzhongyiwei))); replctest = fftshift(fft2(ifftshift(sp_real*Ir)))
cm_ang = (replctest*replcHtest_ang*H_ang_2n)/(spzhongxin*replch_ang + 1e-12)
cm_con = (replctest*replcHtest_con*H_con_2n)/(spzhongxin*replch_con + 1e-12)
angcm = angle(cm_ang); abscm = |cm_con|
phase_vals = angcm[ (replch_ang*H_ang_2n) != 0 ]
mod_vals   = abscm[ (replch_con*H_con_2n) != 0 ]

## Histograms
jdjd = jdc = 0.02
f_bins  = arange(-pi, pi+jdjd, jdjd)
ff_bins = arange(0, 0.6+jdc, jdc)
g_hist  = histogram(real(phase_vals), f_bins)[0]
gg_hist = histogram(real(mod_vals),  ff_bins)[0]

## EMD smoothing (PyEMD EMD, FIXE_H=0)
phase: imf = emd(g_hist); if imf rows>=5 and max(g_hist)<50 -> sum(imf[4:]);
       elif rows>=4 -> sum(imf[3:]); else g_hist. angle6[spi] = -pi + (argmax+1)*jdjd
depth: imf2 = emd(gg_hist); gg_smooth = sum(imf2, axis=0);
       c6[spi] = jdc * mean(where(gg_smooth==max)+1)
Fallback to raw histogram on exception or len<=10.
