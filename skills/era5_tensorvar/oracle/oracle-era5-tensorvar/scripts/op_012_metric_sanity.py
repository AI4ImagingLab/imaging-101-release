#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np


def metrics(estimate, reference, weights):
    eps = 1e-12
    ncc = []
    nrmse = []
    wnrmse = []
    for c in range(estimate.shape[1]):
        e = estimate[:, c].reshape(-1)
        r = reference[:, c].reshape(-1)
        ec = e - e.mean()
        rc = r - r.mean()
        ncc.append(float(np.dot(ec, rc) / (np.linalg.norm(ec) * np.linalg.norm(rc) + eps)))
        nrmse.append(float(np.sqrt(np.mean((e - r) ** 2)) / (r.max() - r.min() + eps)))
        vals = []
        w = weights[c]
        for t in range(estimate.shape[0]):
            vals.append(np.linalg.norm((estimate[t, c] - reference[t, c]) * w) / (np.linalg.norm(reference[t, c] * w) + eps))
        wnrmse.append(float(np.mean(vals)))
    return {'ncc': ncc, 'nrmse': nrmse, 'weighted_nrmse': wnrmse}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--estimate', default='outputs/case_000.npz')
    p.add_argument('--reference', required=True, help='Run-local comparison npz with a state array')
    args = p.parse_args()
    with np.load(args.estimate) as f:
        estimate = f['state']
    with np.load(args.reference) as f:
        reference = f['state']
    with np.load(Path('public/visible_data/cases/case_000/input_data/raw_data.npz')) as f:
        weights = f['lat_weight_matrix'][0]
    print(metrics(estimate, reference, weights))
