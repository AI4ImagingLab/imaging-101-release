"""Maintainer-only analytic synthetic observations, not solver input.

The bound object is a nonnegative difference of two coaxial Gaussians (a broad
1.5 mm component and a narrower 0.7 mm depletion) plus a localized 0.42 mm core.
Widths use exp(-(r/width)^2), not a Gaussian standard-deviation convention.
An analytic infinite-line integral generates projection, independently of the
Hansen-Law discretization. Radius/impact samples are j*0.025 mm; z samples are
(i-32)*0.1 mm. The tail beyond the 6.4 mm detector is negligible but not forced
to zero. Noise is additive independent normal with std 0.002 radiance units.
Default seed 1407 identifies this benchmark acquisition, and is hidden from
solvers. No upstream library is used. See test/validation.md for oracle capture.
"""
import argparse
from pathlib import Path
import numpy as np
from .preprocessing import DR, DZ, NOISE_STD


def synthetic_fields(seed=1407):
    """Return (truth, noisy_projection, clean_projection), all (1,65,257)."""
    r = np.arange(257)*DR
    z = (np.arange(65)-32)*DZ
    widths = np.array([1.5, 0.7, 0.42])
    amplitudes = np.array([0.9*np.exp(-(z/2.3)**2), -0.65*np.exp(-(z/2.3)**2),
                           0.5*np.exp(-((z-1.1)/0.65)**2)]).T
    radial = np.exp(-(r[None, :]/widths[:, None])**2)
    truth = amplitudes @ radial
    clean = amplitudes @ (np.sqrt(np.pi)*widths[:, None]*radial)
    noisy = clean + np.random.default_rng(seed).normal(0, NOISE_STD, clean.shape)
    return truth[None], noisy[None], clean[None]


def generate(destination, seed=1407):
    """Write public observations and hidden truth to a separate destination."""
    destination = Path(destination)
    truth, noisy, _ = synthetic_fields(seed)
    (destination/'data').mkdir(parents=True, exist_ok=True)
    (destination/'evaluation').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination/'data/raw_data.npz', projection=noisy)
    np.savez_compressed(destination/'evaluation/ground_truth.npz', reconstruction=truth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', help='Separate destination; do not overwrite the bound acquisition')
    parser.add_argument('--seed', type=int, default=1407)
    args = parser.parse_args()
    generate(args.destination, args.seed)
