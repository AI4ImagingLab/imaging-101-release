"""Levenberg-Marquardt with autograd column-wise Jacobian.

Run-local helper. forward_fn() returns predicted tensor; residual_fn(y) returns
residual tensor (target - pred) with NaNs zeroed. get/set_param apply updates to
the scene parameter tensors (must keep requires_grad and stay in the graph).
"""
import torch
import torch.autograd.functional as AF


def column_jacobian(forward_fn, inputs):
    """M-by-N Jacobian via column-wise double-grad (efficient for M>>N)."""
    outputs = forward_fn()
    M = outputs.shape
    go = (torch.zeros_like(outputs, requires_grad=True),)
    Js = []
    for x in inputs:
        gi = AF._autograd_grad((outputs,), x, go, create_graph=True)
        N = torch.numel(x)
        if N == 1:
            J = AF._autograd_grad(gi, go, (torch.ones_like(x),), retain_graph=True)[0][..., None]
        else:
            J = torch.zeros((*M, N), device=x.device)
            v = torch.zeros(N, device=x.device)
            for i in range(N):
                v[i] = 1.0
                J[..., i] = AF._autograd_grad(gi, go, (v.view(x.shape),), retain_graph=True)[0]
                v[i] = 0.0
        Js.append(J)
    return torch.cat(Js, dim=-1)


def lm_optimize(forward_fn, residual_fn, params, apply_delta,
                lamb=0.1, mu=2.0, regularization='diag', max_iter=100):
    """params: list of leaf tensors (requires_grad). apply_delta(list_of_delta, sign)
    adds/subtracts deltas in-scene and returns refreshed param list."""
    Ns = [x.numel() for x in params]
    shapes = [list(x.shape) for x in params]
    losses = []
    for it in range(max_iter):
        y = forward_fn()
        with torch.no_grad():
            L = torch.mean(residual_fn(y) ** 2).item()
        if L < 1e-16:
            break
        J = column_jacobian(forward_fn, params).reshape(-1, sum(Ns))
        JtJ = J.T @ J
        n = JtJ.shape[0]
        if regularization == 'I':
            R = torch.eye(n, device=JtJ.device)
        else:
            R = torch.diag(torch.diag(JtJ).abs())
        bb = []
        for x in params:
            g = torch.autograd.grad(y, x, grad_outputs=residual_fn(y), retain_graph=True)[0]
            bb.append(g.reshape(-1))
        b = torch.cat(bb)
        del J
        L_cur = L + 1.0
        inner = 0
        while L_cur >= L:
            inner += 1
            if inner > 20:
                break
            A = JtJ + lamb * R
            try:
                delta = torch.linalg.solve(A, b)
            except Exception:
                delta = torch.linalg.lstsq(A, b).solution
            if torch.isnan(delta).any():
                break
            parts = [d.reshape(s) for d, s in zip(torch.split(delta, Ns), shapes)]
            params = apply_delta(parts, +1)
            with torch.no_grad():
                L_cur = torch.mean(residual_fn(forward_fn()) ** 2).item()
            if L_cur < L:
                lamb /= mu
                break
            lamb *= 2.0 * mu
            params = apply_delta(parts, -1)
            if lamb > 1e16:
                break
        losses.append(L)
        if it > 0 and abs(losses[-2] - L) < 1e-8:
            break
        if torch.mean(torch.abs(delta)).item() < 1e-8:
            break
    return losses, params
