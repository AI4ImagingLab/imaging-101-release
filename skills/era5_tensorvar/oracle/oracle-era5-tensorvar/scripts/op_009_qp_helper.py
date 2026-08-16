#!/usr/bin/env python
import numpy as np
import cvxpy as cp
from cvxpy import quad_form


def solve_latent_qp(z_b, seq_z, F, B, R, Q):
    T, D = seq_z.shape
    pred_seq_z = cp.Variable((T, D))
    penalty = np.linspace(0.1, 1, T - 1)[::-1]
    cost = 0
    for t in range(T):
        if t == 0:
            cost += quad_form(pred_seq_z[t] - z_b, B)
            cost += quad_form(pred_seq_z[t] - seq_z[t], R)
        else:
            cost += quad_form(pred_seq_z[t] - seq_z[t], R)
            diff = pred_seq_z[t] - F.T @ pred_seq_z[t - 1]
            cost += quad_form(diff, penalty[t - 1] * Q)
    cp.Problem(cp.Minimize(cost), []).solve()
    if pred_seq_z.value is None:
        raise RuntimeError('Latent QP did not return a solution')
    return pred_seq_z.value


if __name__ == '__main__':
    print('This helper defines solve_latent_qp(z_b, seq_z, F, B, R, Q).')
