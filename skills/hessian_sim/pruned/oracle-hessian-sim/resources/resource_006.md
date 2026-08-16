# Wiener SIM reconstruction (n_work=512)

## Setup
otfx = 256 if image_size<=256 else 512 if <=512 else image_size  # here 512
zuobiaox_r = zuobiaox*(otfx/n); zuobiaoy_r = zuobiaoy*(otfx/n); n_work=otfx
fc = ceil(200*n_work/512)
otf_r = zoom(otf, n_work/otf.shape[0], order=1) if shape mismatch else otf
jiequ = ones; jiequ[kr>fc]=0  (kr on n_work grid centred)
H = otf_r*jiequ; H /= (H.max()+1e-12); H1=(H!=0)
Hk = pad(H,(2n,2n)); H1_2n = pad(H1,(2n,2n))

## Per-frame precompute (ii in 0..5)
kytest = 2*pi*(zuobiaox_r[ii,0]-n_work)/(2*n_work)
kxtest = 2*pi*(zuobiaoy_r[ii,0]-n_work)/(2*n_work)
Irtest[:,:,ii]   = exp(1j*(kxtest*xx+kytest*yy))
replcHtest[:,:,ii] = |shift_otf(H1_2n,kxtest,kytest,n_work)|
replch[:,:,ii]     = shift_otf(Hk,kxtest,kytest,n_work)
replcHtest = 1 where |.|>0.9 else 0 ; replch *= replcHtest
re = replch; re[re==0]=1e19; re = re/|re|            # phase-only unit
reh = |replch|; hs = sum(reh**2, axis=2)

## Weights & phase offset
xishu = ones(6); for each angle: avg_inv = mean(1/(c6_pair+1e-12));
  xishu[angle*3+1]=xishu[angle*3+2]=avg_inv
deph[angle] = sign(a6_pair[0])*mean(|a6_pair|)

## Reconstruction phase matrix (per angle, from deph)
phi_tmp = linspace(deph, regul+deph, total_spjg+1)
phi = [phi_tmp[0], phi_tmp[spjg[0]], phi_tmp[spjg[0]+spjg[1]]]
pm[j] = [1, exp(1j*phi[j]), exp(-1j*phi[j])]; inv per angle

## Apodization
plong = floor( sum_i sqrt((zx_r[i]-zx_r[0])^2+(zy_r[i]-zy_r[0])^2) / (2*nangles) )
k_max = plong + fc; kr_2n on [-n_work:n_work]^2
bhs = cos(pi*kr_2n/(2*k_max)); bhs[kr_2n>k_max]=0

## Edge taper (sig=0.25)
mask = (sigmoid(sig*x)-sigmoid(sig*(x-sx-1)))*(sigmoid(sig*y)-sigmoid(sig*(y-sy-1)))
mask9 = tile(mask**3 over 6 frames); sigmoid clipped to [-500,500]

## Per time point
fd[:,:,i] = mean over repeats; fd *= mask9
pad to n_work; DIbar = fftshift(fft2(ifftshift(.)))
sp = unmix with inv_phase_matrices per angle; sp *= jiequ
for each band: pad to 2n; retirff = fftshift(fft2(ifftshift(ifft(band)*Irtest)))
retirff *= replcHtest; retirff /= re
tmprc[:,:,t] = xishu[t]*retirff[:,:,t]*conj(reh[:,:,t]) / (hs + 0.005*nangles*weilac**2)
dr = sum(tmprc,axis=2); drr = dr*bhs
fimage = fftshift(ifft2(ifftshift(drr)))
result = real(fimage[n_work-sy:n_work+sy, n_work-sx:n_work+sx]); result[result<0]=0
append float32; advance by nframes*beishu_re
