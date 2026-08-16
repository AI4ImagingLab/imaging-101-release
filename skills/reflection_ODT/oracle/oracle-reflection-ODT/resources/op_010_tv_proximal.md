Chambolle TV proximal details

For each 2D slice independently:

1. Start dual arrays `px=0`, `py=0`.
2. Repeat 20 times:
   - `div_p = divergence(px, py)`
   - `x_hat = img - tau*div_p`
   - compute forward differences of `x_hat`
   - `px_new = px + gx/(8*tau)` and `py_new = py + gy/(8*tau)`
   - `norm = sqrt(px_new^2 + py_new^2)` clipped below by 1
   - `px = px_new/norm`, `py = py_new/norm`
3. Return `img - tau*divergence(px,py)`.

This is 2D TV per slice; do not include finite differences along z.