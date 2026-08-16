# Richardson-Lucy loop

Constants: iterations = 10, eps = 1e-8.

```
observation = asarray(observation, float64)
recon = ones(tex_shape + (n_depths,), float64)          # all-ones init
norm_bp = backward_project(ones_like(observation))
norm_bp = maximum(norm_bp, eps)
for q in range(iterations):
    estimate   = forward_project(recon)
    ratio      = observation / maximum(estimate, eps)
    correction = backward_project(ratio) / norm_bp
    correction[~isfinite(correction)] = 0; correction[correction<0] = 0
    recon *= correction
    recon[~isfinite(recon)] = 0; recon[recon<0] = 0
return recon                                            # (450,450,n_depths)
```

Reconstructed target image for the case = recon[:, :, 0].
Update rule:  v <- v * H^T(y / (H v + eps)) / H^T(1).