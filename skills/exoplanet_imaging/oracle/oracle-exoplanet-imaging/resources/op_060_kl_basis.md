# KL basis (PSF subspace) formulas

Reference matrix R: shape (N, n_pix), mean-subtracted frames, NaN replaced by 0.

## SVD (default)
    U, S, Vt = svd(R, full_matrices=False)   # Vt: (min(N,n_pix), n_pix)
    basis = Vt.T[:, :K_max]                   # (n_pix, K_max)

## Low-rank PCA
    _, _, V = pca_lowrank(R, q=K_max, center=False)
    basis = V                                 # (n_pix, K_max)

## Eigendecomposition (Soummer, Pueyo & Larkin 2012, Eq.5)
    cov = R @ R.T                             # (N, N)
    eigvals, eigvecs = eigh(cov)              # ascending
    eigvals = eigvals.flip(0)                 # descending
    eigvecs = eigvecs.flip(1)
    scales  = (1.0 / sqrt(eigvals[:K_max]))   # (K_max,)
    basis   = R.T @ (eigvecs[:, :K_max] * scales[None, :])   # (n_pix, K_max)

All three yield equivalent orthonormal KL eigenimages up to sign. Compute at K_max = max(K list) once, then slice [:, :k].

Typical K: 5-20 for ~60 frames; standard run uses K=10 and K=20.
