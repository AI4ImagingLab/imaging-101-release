#!/usr/bin/env python
"""Generate the preprocessing test fixtures for seismic_FWI_original.

The evaluation/fixtures/ directory is gitignored repo-wide, so fixtures must be
regenerated locally before running the tests. This script (re)creates the
deterministic fixtures consumed by evaluation/tests/test_preprocessing.py:

    input_velocity.npy                 (60, 40) float32  – a Marmousi v_true crop
    output_preprocess_velocity_f3.npy  golden preprocess_velocity(input, factor=3)[0]
    output_initial_model_s8.npy        golden make_initial_model(input, sigma=8)

The input is a fixed crop of the real Marmousi model (data/ground_truth.npz),
so the fixtures are fully deterministic. Existing physics_model/solver fixtures
are not touched.

Usage
-----
    cd tasks/seismic_FWI_original
    python generate_fixtures.py
"""

import os
import sys
import numpy as np
import torch

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import preprocess_velocity, make_initial_model

FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures')
DATA_DIR = os.path.join(TASK_DIR, 'data')


def main():
    os.makedirs(FIX_DIR, exist_ok=True)

    v_true = np.load(os.path.join(DATA_DIR, 'ground_truth.npz'))['v_true']
    v_in = np.ascontiguousarray(v_true[:60, :40]).astype(np.float32)
    np.save(os.path.join(FIX_DIR, 'input_velocity.npy'), v_in)

    v_t = torch.from_numpy(v_in)

    v_red, dx = preprocess_velocity(v_t, factor=3)
    np.save(os.path.join(FIX_DIR, 'output_preprocess_velocity_f3.npy'),
            v_red.numpy())

    v_init = make_initial_model(v_t, sigma=8.0)
    np.save(os.path.join(FIX_DIR, 'output_initial_model_s8.npy'),
            v_init.numpy())

    print(f"input_velocity            {v_in.shape}")
    print(f"output_preprocess (f=3)   {tuple(v_red.shape)}  dx={dx}")
    print(f"output_initial_model (s=8){tuple(v_init.shape)}")
    print(f"Wrote fixtures to {FIX_DIR}")


if __name__ == '__main__':
    main()
