"""Dispersion shift operators for single-disperser CASSI (step=1).

Run with the run-local runtime interpreter, e.g.:
    public/envs/runtime/.venv-posix/bin/python -c "import op_030_dispersion"
"""
import numpy as np


def shift(inputs, step=1):
    """(H, W, nC) -> (H, W+(nC-1)*step, nC) dispersed cube."""
    row, col, nC = inputs.shape
    output = np.zeros((row, col + (nC - 1) * step, nC), dtype=inputs.dtype)
    for i in range(nC):
        output[:, i * step:i * step + col, i] = inputs[:, :, i]
    return output


def shift_back(inputs, step=1):
    """(H, W+(nC-1)*step, nC) -> (H, W, nC). Mutates `inputs` in place (np.roll)."""
    row, col, nC = inputs.shape
    for i in range(nC):
        inputs[:, :, i] = np.roll(inputs[:, :, i], (-1) * step * i, axis=1)
    return inputs[:, 0:col - step * (nC - 1), :]
