"""Notebook-only correlated velocity perturbation diagnostic.

The bound real raw_data.npz is never regenerated or modified. A seed-controlled
zero-mean Gaussian velocity perturbation has covariance fraction² Cd, where Cd
is the legitimate observation covariance in (mm/year)². Boundary convention is
the finite ordered vector: three SAR blocks, GNSS east, GNSS north; no padding,
interpolation or periodic boundary is applied. No hidden slip field is read.
"""
import numpy as np


def correlated_velocity_perturbation(data, covariance, seed=20261022, fraction=.15):
    """Return copied observation dict and perturbation3554 in mm/year."""
    if not np.isfinite(fraction) or fraction<0:
        raise ValueError('Perturbation fraction must be finite and nonnegative')
    c=np.asarray(covariance)
    if c.shape!=(3554,3554) or not np.isfinite(c).all():
        raise ValueError('Expected finite3554x3554 velocity covariance')
    noise=fraction*(np.linalg.cholesky(c)@np.random.default_rng(seed).standard_normal(3554))
    result={k:v.copy() for k,v in data.items()}
    result['sar_velocity']+=noise[:3476]
    result['gnss_velocity']+=noise[3476:].reshape(2,39).T
    return result,noise
