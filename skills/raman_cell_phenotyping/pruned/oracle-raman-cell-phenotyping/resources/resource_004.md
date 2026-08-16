# FCLS per pixel (Heinz 2001), scipy SLSQP

```
fcls_pixel(y, E):                     # y:(B',)  E:(K,B')
    K = E.shape[0]
    objective = lambda a: 0.5 * dot(y - a @ E, y - a @ E)
    gradient  = lambda a: -E @ (y - a @ E)
    constraints = {type:'eq', fun: lambda a: sum(a) - 1.0, jac: lambda a: ones(K)}
    bounds = [(0, None)] * K
    a0 = ones(K) / K
    res = minimize(objective, a0, jac=gradient, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={maxiter:200, ftol:1e-12})
    return res.x

estimate_abundances_fcls(flat, E):    # flat:(N,B')
    return stack([fcls_pixel(flat[i], E) for i in range(N)])   # (N, K)
```

The non-negativity bounds plus the sum-to-one equality make abundances physically interpretable fractional compositions. Note the model uses a @ E (a is a row of length K, E is (K,B')).