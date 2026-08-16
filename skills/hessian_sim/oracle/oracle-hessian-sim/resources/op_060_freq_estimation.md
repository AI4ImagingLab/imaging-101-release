# Pattern frequency estimation

## Support (fc = ceil(220*n/512))
H = |otf with kr>fc zeroed|; H1 = (H!=0); H2 = H1
H_2n = pad(H,(2n,2n)); H1_2n = pad(H1,(2n,2n))

## Loop indices (spi in range(0, nangles*(nphases-1), 2); spi_m = spi+1)
idx_center  = ceil(spi_m/2)*nphases - 1 - 1   (0-based)
idx_shifted = ceil(spi_m/2) + 2*floor(spi_m/2) - 1
store idx1  = spi_m + 1 + floor((spi_m-1)/2) - 1 ; idx2 = idx1+1
zuobiaox[idx1]=maxx; zuobiaoy[idx1]=maxy; zuobiaox[idx2]=2n-maxx; zuobiaoy[idx2]=2n-maxy

## Coarse cross-correlation
cishu = dft_conv(H2, H2[::-1,::-1]); ci = 1 where |cishu|>=0.9 else 0
jieguo = dft_conv(sp_center, conj(sp_shifted)[::-1,::-1]) * ci
lihe = |jieguo / (cishu + 1e-12)|
restrict lihe to annulus (pg-fanwei <= kr <= pg+fanwei) on the conv-size grid; argmax -> (maxx,maxy)

## 2n mapping
kx = 2*pi*(maxy - n)/(2n); ky = 2*pi*(maxx - n)/(2n)

## Merit (_compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n, kx, ky, n))
replcHtest = shift_otf(H1_2n, kx, ky, n); mask -> 1 where |.|>0.9 else 0
replch     = shift_otf(H_2n, kx, ky, n) * replcHtest
sp_shifted_real = fftshift(ifft2(ifftshift(sp_shifted_2n)))
Ir = exp(1j*(kx*xx + ky*yy)); replctest = fftshift(fft2(ifftshift(sp_shifted_real*Ir)))
youhua = replctest * replcHtest * H_2n
num = |sum(conj(youhua) * (sp_center_2n * replch))|
den = sum(H1_2n * replcHtest)
merit = num / (den + 1e-12)

## Coordinate descent
step_x=step_y=1.0; eps=1e-5. Probe +/-eps on the axis being optimised; move in improving
direction by flag*step; on no improvement halve step; stop when both steps < 1e-4.
Accept trial only if merit strictly increases.
