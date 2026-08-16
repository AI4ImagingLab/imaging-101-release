# Aspheric sag reference

Sag (z as function of squared radius r2 = x^2 + y^2):

    sag(r2) = (c*r2) / (1 + sqrt(1 - (1+k)*c^2*r2)) + sum_i ai[i] * r2^(i+2)

First derivative w.r.t. r2:

    alpha = (1+k)*c^2*r2
    s = sqrt(1 - alpha)
    d_sag/d_r2 = c*(1 + s - 0.5*alpha)/(s*(1+s)^2) + sum_i (i+2)*ai[i]*r2^(i+1)

Normal (unit): normalize([2*(d_sag/d_r2)*x, 2*(d_sag/d_r2)*y, -1]).

Notes:
- c = 1/roc. Sign of c encodes surface orientation; the recovered roc = 1/c can be negative.
- For a plain sphere set k=0, ai=None.
- Keep everything differentiable; do not detach c or thickness.