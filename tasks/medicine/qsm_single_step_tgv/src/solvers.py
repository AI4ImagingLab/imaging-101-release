"""Local Chambolle--Pock SSTGV, including the anisotropic-adjoint correction.

QSM.m (c) 2020 kamesy, MIT. See test/validation.md for the full notice,
the original numerical oracle, and the separately identified correction.
"""
import time
import numpy as np
from .physics_model import (wrapped_laplacian, interior_mask, forward_difference,
    backward_difference, laplacian, wave_operator, symmetric_gradient,
    symmetric_divergence, project_balls)
from .preprocessing import SPACING, PHASE_PER_PPM, crop_slices

DEFAULT_LAMBDA = PHASE_PER_PPM*0.0015/2
DEFAULT_RATIO = 1/3


def step_sizes(spacing):
    """Return the source's conservative shared primal/dual step size."""
    h = np.asarray(spacing)
    g = 4*np.sum(1/h**2)
    v = 2*(1/h[0]**2+1/h[1]**2+2/h[2]**2)
    return 1/np.sqrt(g*g+v*v)


def primal_dual(field_rhs, mask, spacing=SPACING, lam=DEFAULT_LAMBDA, ratio=DEFAULT_RATIO,
                tolerance=3e-4, max_iterations=5000, return_state=False):
    """Solve the cropped phase-unit constrained TGV problem.

    field_rhs is minus the wrapped phase Laplacian, rad/mm^2. All state
    arrays are float64; mask is boolean. Stop is checked every ten updates
    against the previous SINGLE update, as in the original implementation.
    """
    if field_rhs.ndim != 3 or field_rhs.shape != mask.shape or min(mask.shape) < 3:
        raise ValueError('Expected matching three-dimensional RHS and mask')
    if (not np.isfinite(field_rhs).all() or lam <= 0 or ratio < 0
            or max_iterations < 1 or tolerance < 0):
        raise ValueError('Invalid solver parameters')
    m = mask.astype(bool).copy()
    m[[0, -1], :, :] = False
    m[:, [0, -1], :] = False
    m[:, :, [0, -1]] = False
    mi = interior_mask(mask)
    dt = step_sizes(spacing)
    scalar = lambda: np.zeros_like(field_rhs, dtype=np.float64)
    chi, eta, nu = scalar(), scalar(), scalar()
    w = np.zeros((3, *mask.shape))
    p, q = np.zeros_like(w), np.zeros((6, *mask.shape))
    cb, eb, wb = chi.copy(), eta.copy(), w.copy()
    history = []
    for iteration in range(1, max_iterations+1):
        old_chi, old_eta = chi, eta
        nu += dt*(laplacian(eb, spacing)-wave_operator(cb, spacing)+field_rhs)*mi
        p = project_balls(p + dt*(np.array([forward_difference(cb, a, spacing)
                           for a in range(3)])-wb)*mi, lam)
        if ratio > 0:
            q = project_balls(q + dt*symmetric_gradient(wb, spacing)*mi, ratio*lam, tensor=True)
        eta = ((eta-dt*laplacian(nu, spacing))/(1+dt))*m
        chi = (chi+dt*(sum(backward_difference(p[a], a, spacing)
                    for a in range(3))+wave_operator(nu, spacing)))*m
        cb, eb = 2*chi-old_chi, 2*eta-old_eta
        if ratio > 0:
            wn = (w+dt*(p+symmetric_divergence(q, spacing)))*m
            wb = 2*wn-w
            w = wn
        if iteration % 10 == 0:
            nc, ne = np.linalg.norm(chi.ravel()), np.linalg.norm(eta.ravel())
            dc = np.linalg.norm((chi-old_chi).ravel())/nc if nc else np.inf
            de = np.linalg.norm((eta-old_eta).ravel())/ne if ne else np.inf
            history.append((iteration, dc, de))
            if dc < tolerance and de < 2*tolerance:
                break
    result = {'chi': chi, 'eta': eta, 'iterations': iteration, 'history': np.array(history).reshape(-1, 3)}
    if return_state:
        result.update(eta=eta, w=w, nu=nu, p=p, q=q, interior=mi)
    return result


def reconstruct(phase, mask, spacing=SPACING, lam=DEFAULT_LAMBDA, ratio=DEFAULT_RATIO,
                tolerance=3e-4, max_iterations=5000, device='cpu'):
    """Actively reconstruct the full observation; return ppm array and diagnostics."""
    start = time.perf_counter()
    crop = crop_slices(mask)
    rhs = -wrapped_laplacian(phase[crop], spacing)
    if device == 'cpu':
        result = primal_dual(rhs, mask[crop], spacing, lam, ratio, tolerance, max_iterations, return_state=True)
    else:
        result = primal_dual_torch(rhs, mask[crop], spacing, lam, ratio,
                                  tolerance, max_iterations, return_state=True, device=device)
    full = np.zeros_like(phase, dtype=np.float64)
    full[crop] = result.pop('chi')/PHASE_PER_PPM
    residual_phase = np.zeros_like(phase, dtype=np.float64)
    residual_phase[crop] = result.pop('eta')
    auxiliary_w = np.zeros((3,*phase.shape),dtype=np.float64)
    auxiliary_w[(slice(None),*crop)] = result.pop('w')/PHASE_PER_PPM
    for key in ['nu','p','q','interior']:result.pop(key,None)
    result.update(runtime_seconds=time.perf_counter()-start, crop=crop, residual_phase=residual_phase, auxiliary_w=auxiliary_w)
    return full, result


def primal_dual_torch(field_rhs, mask, spacing=SPACING, lam=DEFAULT_LAMBDA, ratio=DEFAULT_RATIO,
                      tolerance=3e-4, max_iterations=5000, return_state=False,
                      device='cuda'):
    """Same explicit updates on PyTorch tensors; double precision throughout.

    ``device='cpu'`` permits independent CPU/backend parity on small fixtures.
    No automatic device fallback changes the declared reconstruction path.
    """
    import torch
    if field_rhs.ndim != 3 or field_rhs.shape != mask.shape or min(mask.shape) < 3:
        raise ValueError('Expected matching three-dimensional RHS and mask')
    if (not np.isfinite(field_rhs).all() or lam <= 0 or ratio < 0
            or max_iterations < 1 or tolerance < 0):
        raise ValueError('Invalid solver parameters')
    f = torch.as_tensor(np.ascontiguousarray(field_rhs), dtype=torch.float64, device=device)
    m = torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.bool, device=device).clone()
    m[[0, -1], :, :] = False
    m[:, [0, -1], :] = False
    m[:, :, [0, -1]] = False
    mi = torch.as_tensor(interior_mask(mask), device=device)
    h = tuple(float(x) for x in spacing)
    dt = step_sizes(h)
    chi, eta, nu = torch.zeros_like(f), torch.zeros_like(f), torch.zeros_like(f)
    w = torch.zeros((3, *mask.shape), dtype=f.dtype, device=device)
    p, q = torch.zeros_like(w), torch.zeros((6, *mask.shape), dtype=f.dtype, device=device)
    cb, eb, wb = chi.clone(), eta.clone(), w.clone()
    history = []

    def difference(u, axis, forward=True):
        return ((torch.roll(u, -1, axis)-u) if forward else
                (u-torch.roll(u, 1, axis)))/h[axis]

    def second_order(u, wave=False):
        coefficients = (-1/3, -1/3, 2/3) if wave else (1, 1, 1)
        return sum(c*(torch.roll(u, -1, a)-2*u+torch.roll(u, 1, a))/h[a]**2
                   for a, c in enumerate(coefficients))

    def project(v, radius, tensor=False):
        norm2 = v[0]**2+v[3]**2+v[5]**2+2*(v[1]**2+v[2]**2+v[4]**2) if tensor else (v*v).sum(0)
        return v/torch.clamp(torch.sqrt(norm2)/radius, min=1).unsqueeze(0)

    with torch.no_grad():
        for iteration in range(1, max_iterations+1):
            old_chi, old_eta = chi, eta
            nu += dt*(second_order(eb)-second_order(cb, True)+f)*mi
            gradient = torch.stack([difference(cb, a) for a in range(3)])
            p = project(p+dt*(gradient-wb)*mi, lam)
            if ratio > 0:
                ew = torch.stack([difference(wb[0], 0),
                    (difference(wb[1], 0)+difference(wb[0], 1))/2,
                    (difference(wb[2], 0)+difference(wb[0], 2))/2,
                    difference(wb[1], 1),
                    (difference(wb[2], 1)+difference(wb[1], 2))/2,
                    difference(wb[2], 2)])
                q = project(q+dt*ew*mi, ratio*lam, True)
            eta = ((eta-dt*second_order(nu))/(1+dt))*m
            chi = (chi+dt*(sum(difference(p[a], a, False) for a in range(3))
                           +second_order(nu, True)))*m
            if ratio > 0:
                divq = torch.stack([
                    difference(q[0], 0, False)+difference(q[1], 1, False)+difference(q[2], 2, False),
                    difference(q[1], 0, False)+difference(q[3], 1, False)+difference(q[4], 2, False),
                    difference(q[2], 0, False)+difference(q[4], 1, False)+difference(q[5], 2, False)])
                wn = (w+dt*(p+divq))*m
                wb = 2*wn-w
                w = wn
            cb, eb = 2*chi-old_chi, 2*eta-old_eta
            if iteration % 10 == 0:
                nc, ne = torch.linalg.vector_norm(chi), torch.linalg.vector_norm(eta)
                dc = float(torch.linalg.vector_norm(chi-old_chi)/nc) if nc else float('inf')
                de = float(torch.linalg.vector_norm(eta-old_eta)/ne) if ne else float('inf')
                history.append((iteration, dc, de))
                if dc < tolerance and de < 2*tolerance:
                    break
    result = {'chi': chi.cpu().numpy(), 'eta': eta.cpu().numpy(), 'iterations': iteration,
              'history': np.array(history).reshape(-1, 3)}
    if return_state:
        result.update({k: v.cpu().numpy() for k, v in
            dict(eta=eta, w=w, nu=nu, p=p, q=q, interior=mi).items()})
    return result
