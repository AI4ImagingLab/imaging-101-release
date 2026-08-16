# ASLS baseline (Eilers & Boelens 2005), per spectrum along last axis

Parameters: lam = 1e6, p = 1e-2, max_iter = 50, tol = 1e-3.

```
asls_single(spectrum, lam, p, max_iter, tol):
    m = len(spectrum)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(m-2, m), format='csc')  # 2nd-order diff
    DTD = lam * D.T.dot(D)
    w = ones(m)
    z = spectrum.copy()
    for _ in range(max_iter):
        W = sparse.diags(w, 0, shape=(m, m), format='csc')
        z_new = spsolve(W + DTD, w * spectrum)
        if norm(z_new - z) / (norm(z) + 1e-30) < tol:
            z = z_new; break
        z = z_new
        w = where(spectrum > z, p, 1 - p)
    return z   # baseline
```

Corrected data = intensity - baseline, computed with an along-last-axis map over the volume. Note the convergence test runs before the weight update, so the first pass uses w = ones.