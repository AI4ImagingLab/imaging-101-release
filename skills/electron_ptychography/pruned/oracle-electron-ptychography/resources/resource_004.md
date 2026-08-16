# DPC reference

Fourier gradient integration operators (solve grad(phi) ~ (gx,gy)):
```
kx=fftfreq(Nx,d=dx); ky=fftfreq(Ny,d=dy); KY,KX=meshgrid(ky,kx)
k2=KX**2+KY**2; k2[0,0]=inf
phi_update = real( ifft2( fft2(gx)*(-1j*0.25*KX/k2) + fft2(gy)*(-1j*0.25*KY/k2) ) )
```
Iterative loop with step halving:
```
phase=0; step=0.5; prev=inf
for it in range(64):
    dphx=(roll(phase,1,0)-roll(phase,-1,0))/(2*dx)
    dphy=(roll(phase,1,1)-roll(phase,-1,1))/(2*dy)
    rx=dphx-gx; ry=dphy-gy
    err=mean(rx**2+ry**2)
    if err>prev: step/=2; if step<1e-6: break
    prev=err
    phase += step*fourier_integrate(rx,ry,dx,dy)
```
BF mask threshold 0.8 of dp_mean (reference used dp_mean>0.8 directly). CoM mean-subtracted before rotation.