#!/usr/bin/env python
"""Angular-sensitivity alpha(u,v) models for FLFM parallax depth coding.

compute_alpha(uv, model, na, n, uv_scaling, m=10) -> (N,2) alpha array.
- LINEAR:           alpha = uv
- SPHERE:           alpha = -(na/n)/sqrt(1 - rho^2 (na/n)^2) * uv
- INTEGRATE_SPHERE: phase-average SPHERE over each microlens aperture (width uv_scaling)

Run from the eval run root; pure numpy.
"""
import warnings
import numpy as np


def compute_alpha(uv, model, na, n, uv_scaling=None, m=10):
    uv = np.asarray(uv, dtype=float)
    na_n = na / n
    if model == "LINEAR":
        return uv.copy()
    if model == "SPHERE":
        rho = np.sqrt(np.sum(uv ** 2, axis=1))
        dr = 1.0 - rho ** 2 * na_n ** 2
        dr = np.where(dr < 0.0, np.nan, dr)
        phi = -na_n / np.sqrt(dr)
        return uv * phi[:, None]
    if model == "INTEGRATE_SPHERE":
        if uv_scaling is None:
            raise ValueError("uv_scaling (mla_to_uv_scale) required for INTEGRATE_SPHERE")
        ds2 = uv_scaling / 2.0
        g = np.linspace(-ds2, ds2, m + 1)
        uu = uv[:, 0, None, None] + g[None, None, :]
        vv = uv[:, 1, None, None] + g[None, :, None]
        dr = 1.0 - (uu ** 2 + vv ** 2) * na_n ** 2
        dr = np.where(dr < 0.0, np.nan, dr)
        phi = -na_n / np.sqrt(dr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            au = np.nanmean(uu * phi, axis=(1, 2))
            av = np.nanmean(vv * phi, axis=(1, 2))
        return np.column_stack((au, av))
    raise ValueError(f"Unknown alpha model {model!r}")


if __name__ == "__main__":
    uv = np.array([[0.0, 0.0], [0.6452, 0.0090], [0.3304, -0.5542]])
    print(compute_alpha(uv, "INTEGRATE_SPHERE", 1.27, 1.33, uv_scaling=0.645219, m=10))
