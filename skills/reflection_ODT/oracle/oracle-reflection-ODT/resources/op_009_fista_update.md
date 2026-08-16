Default solver settings

- `n_iter = 50`
- `lr = 5.0`
- `tv_weight = 8e-7`
- `positivity = False`
- Initial `delta_n = 0`
- Initial `x_prev = 0`

FISTA recurrence:

```
q(0) = 1
q(k) = (1 + sqrt(1 + 4*q(k-1)^2))/2
beta_k = (q(k)-1)/q(k+1)
```

Update sequence:

```
x_k = delta_n - lr * gradient
x_k = prox_tv(x_k, lr*tv_weight)
delta_n = x_k + beta_k * (x_k - x_prev)
x_prev = x_k
```

If runtime is constrained, fewer iterations can produce a valid file, but the faithful setting is 50 iterations.