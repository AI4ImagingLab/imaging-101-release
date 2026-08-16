# GAP iteration skeleton (dispersed frame + scene-frame prior)

```
x = At(y, Phi)          # init
y1 = zeros_like(y)
Phi_sum = sum(Phi, axis=2); Phi_sum[Phi_sum==0] = 1
k = 0
for each iteration:
    yb = A(x, Phi)
    y1 = y1 + (y - yb)                      # accelerate
    x  = x + lambda * At((y1 - yb)/Phi_sum, Phi)
    x  = shift_back(x, step=1)              # -> scene frame (H, W, nC)
    x  = PRIOR(x)                           # TV or learned denoiser
    x  = shift(x, step=1)                   # -> dispersed frame
    k += 1
```

Notes:
- lambda = 1, accelerate = True.
- The prior operates in scene coordinates (256x256x31); the data step operates in dispersed coordinates (256x286x31).
- After the final iteration, apply one more shift_back(step=1) to obtain the deliverable cube.
