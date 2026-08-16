# Centred FFT + wavevector mapping

Forward spectrum: fftshift(fft2(ifftshift(x)))
Inverse image:    fftshift(ifft2(ifftshift(X)))

## Coordinate to angular frequency (2n grid)
Given stored pattern pixel coords (zx, zy) on the 2n grid:
    ky = 2*pi*(zx - n)/(2*n)
    kx = 2*pi*(zy - n)/(2*n)
Then linear-phase ramp Ir = exp(1j*(kx*xx + ky*yy)) with xx,yy = meshgrid(arange(2n), arange(2n)).

Applying Ir in real space before FFT shifts a band by (kx,ky). The default (no shift) coordinate is (n, n) i.e. kx=ky=0.
