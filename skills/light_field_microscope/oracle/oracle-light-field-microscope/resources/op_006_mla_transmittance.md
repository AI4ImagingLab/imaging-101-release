# MLA phase mask and tiling

## Single-lenslet pattern (local nnum grid)
```
yy, xx = meshgrid(y_ml_space, x_ml_space, indexing='ij')
phase  = exp(-1j * k/(2*fm) * (yy**2 + xx**2))
if uLensMask == 1:
    phase = where(sens_mask, phase, 0)
else:
    r = sqrt(yy**2+xx**2); phase = where(r < lensPitch/2 - 3, phase, 0)
```

## Lenslet centers (regular grid)
```
coords = arange(n_lenslets)*newSpacingPx
coords = coords - coords[n_lenslets//2]
rows, cols = meshgrid(coords, coords, indexing='ij')
```
Use only centers within radius (psf_size_lenslets + 3) of the grid center.

## Tiling
canvas shape = (len(yspace)+2*len(y_ml_space), len(xspace)+2*len(x_ml_space)), complex.
For each used center offset, ADD the single-lenslet pattern centered at (canvas_center_row+row_off, canvas_center_col+col_off), clipping at canvas edges.
Crop the canvas back to (len(yspace), len(xspace)) about its center -> mla_array.

## Post-MLA field
U_plus = U_mla_field * mla_array   (elementwise complex)