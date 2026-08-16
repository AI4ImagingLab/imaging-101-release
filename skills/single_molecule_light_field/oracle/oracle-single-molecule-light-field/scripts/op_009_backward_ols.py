#!/usr/bin/env python
"""OLS backward model for multi-view FLFM parallax inversion.

get_backward_model(locs, rho_scaling) -> (model[3], std_err[3], mse)
locs is (K,>=12): col1,2=u,v ; col3,4=x,y ; col10,11=alpha_u,alpha_v.
Pure numpy; run from the eval run root.
"""
import numpy as np


def get_backward_model(locs, rho_scaling):
    u = locs[:, 1]
    v = locs[:, 2]
    x = locs[:, 3] - u / rho_scaling
    y = locs[:, 4] - v / rho_scaling
    alpha_u = locs[:, 10]
    alpha_v = locs[:, 11]
    ones = np.ones_like(alpha_u)
    zeros = np.zeros_like(alpha_u)
    A = np.row_stack((
        np.column_stack((ones, zeros, alpha_u)),
        np.column_stack((zeros, ones, alpha_v)),
    ))
    b = np.concatenate((x, y))
    model, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    mse = (np.sum(residuals) / (A.shape[0] - rank)) if residuals.size > 0 else 0.0
    cov = mse * np.linalg.inv(A.T @ A)
    std_err = np.sqrt(np.diag(cov))
    return model, std_err, mse


def forward_model_error(model, points, rho_scaling):
    """Per-view residual (K,2): xy - uv/rho - xy0 - z*alpha."""
    uv = points[:, 1:3]
    xy = points[:, 3:5]
    alpha_uv = points[:, 10:12]
    return xy - (uv / rho_scaling) - model[0:2] - (model[2] * alpha_uv)
