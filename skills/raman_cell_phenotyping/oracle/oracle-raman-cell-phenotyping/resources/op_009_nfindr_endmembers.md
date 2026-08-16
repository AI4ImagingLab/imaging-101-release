# N-FINDR (Winter 1999), numpy + sklearn PCA

```
simplex_volume(reduced):            # reduced shape (K, K-1)
    K = reduced.shape[0]
    M = column_stack([ones(K), reduced])   # (K, K)
    return abs(det(M))

extract_endmembers_nfindr(flat, K, max_iter=50, rng):
    N, B = flat.shape
    data_reduced = PCA(n_components=K-1).fit_transform(flat)   # (N, K-1)
    endmember_indices = rng.choice(N, size=K, replace=False)   # distinct
    for _ in range(max_iter):
        changed = False
        current = data_reduced[endmember_indices]
        current_vol = simplex_volume(current)
        for pixel_idx in range(N):
            for k in range(K):
                if pixel_idx == endmember_indices[k]: continue
                trial = current.copy(); trial[k] = data_reduced[pixel_idx]
                v = simplex_volume(trial)
                if v > current_vol:
                    endmember_indices[k] = pixel_idx
                    current = data_reduced[endmember_indices]
                    current_vol = v
                    changed = True
        if not changed: break
    return flat[endmember_indices].copy()   # (K, B')
```

Notes: PCA reduces to exactly K-1 dims so the K points define a (K-1)-simplex. Volume via |det| of the ones-augmented matrix. The replacement scan updates current/current_vol in place as soon as an improvement is found (greedy within a pass). Convergence when a whole pass yields no replacement, or after max_iter passes.