"""
Optimization solvers for differentiable refractive deflectometry.

Implements Levenberg-Marquardt with autograd-based Jacobian computation.

Extracted from: diffmetrology/solvers.py (Optimization, LM classes) and
diffmetrology/utils.py (DiffMetrology.solve, change_parameters).
"""
import time

import numpy as np
import torch
import torch.autograd.functional as F

from .physics_model import PrettyPrinter, get_nested_attr, set_nested_attr


# ===========================================================================
# Parameter management (replaces exec()-based access in utils.py)
# ===========================================================================

def setup_diff_parameters(scene, param_names, device=torch.device('cpu')):
    """Enable gradients on specified scene parameters.

    Args:
        scene: Scene object
        param_names: list of dot-separated parameter paths (e.g. 'lensgroup.surfaces[0].c')

    Returns:
        list of parameter tensors with requires_grad=True
    """
    diff_parameters = []
    for name in param_names:
        param = get_nested_attr(scene, name)
        if not param.requires_grad:
            param = param.detach()
            param.requires_grad = True
            set_nested_attr(scene, name, param)
        diff_parameters.append(get_nested_attr(scene, name))
    return diff_parameters


def change_parameters(scene, param_names, deltas, sign=True):
    """Apply parameter updates to scene, returning the new parameter tensors.

    Args:
        scene: Scene object
        param_names: list of dot-separated parameter paths
        deltas: list of delta tensors
        sign: True to add deltas, False to subtract

    Returns:
        list of updated parameter tensors
    """
    diff_parameters = []
    for name, delta in zip(param_names, deltas):
        current = get_nested_attr(scene, name)
        if sign:
            new_val = current + delta
        else:
            new_val = current - delta
        set_nested_attr(scene, name, new_val)
        diff_parameters.append(get_nested_attr(scene, name))
    return diff_parameters


# ===========================================================================
# Levenberg-Marquardt solver (from solvers.py)
# ===========================================================================

class LMSolver:
    """Levenberg-Marquardt optimizer with autograd Jacobians.

    Args:
        lamb: initial damping factor
        mu: damping rate (>1)
        regularization: 'I' (identity), 'diag' (diagonal of JtJ), or a vector
        max_iter: maximum LM iterations
    """

    def __init__(self, lamb=0.1, mu=2.0, regularization='diag', max_iter=100):
        self.lamb = lamb
        self.mu = mu
        self.regularization = regularization
        self.max_iter = max_iter

    def jacobian(self, func, inputs, create_graph=False, strict=False):
        """Construct M-by-N Jacobian matrix via column-wise evaluation (forward-mode AD).

        For tall Jacobians (M >> N), column-wise evaluation is more efficient.
        Modified from torch.autograd.functional.jvp().
        """
        Js = []
        outputs = func()
        M = outputs.shape

        grad_outputs = (torch.zeros_like(outputs, requires_grad=True),)
        for x in inputs:
            grad_inputs = F._autograd_grad(
                (outputs,), x, grad_outputs, create_graph=True
            )

            F._check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

            N = torch.numel(x)
            if N == 1:
                J = F._autograd_grad(
                    grad_inputs, grad_outputs, (torch.ones_like(x),),
                    create_graph=create_graph,
                    retain_graph=True
                )[0][..., None]
            else:
                J = torch.zeros((*M, N), device=x.device)
                v = torch.zeros(N, device=x.device)
                for i in range(N):
                    v[i] = 1.0
                    J[..., i] = F._autograd_grad(
                        grad_inputs, grad_outputs, (v.view(x.shape),),
                        create_graph=create_graph,
                        retain_graph=True
                    )[0]
                    v[i] = 0.0
            Js.append(J)
        return torch.cat(Js, axis=-1)

    def optimize(self, forward_fn, scene, param_names, residual_fn, device=torch.device('cpu')):
        """Run LM optimization.

        Args:
            forward_fn: callable returning predicted observations tensor
            scene: Scene object containing the parameters
            param_names: list of parameter name strings
            residual_fn: callable(predicted) -> residual tensor (y_ref - y)
            device: torch device

        Returns:
            loss_history: list of loss values per iteration
        """
        print('optimizing ...')
        time_start = time.time()

        diff_parameters = setup_diff_parameters(scene, param_names, device)
        Ns = [x.numel() for x in diff_parameters]
        NS = [[*x.shape] for x in diff_parameters]

        ls = []
        lamb = self.lamb

        with torch.autograd.set_detect_anomaly(False):
            for it in range(self.max_iter):
                y = forward_fn()
                with torch.no_grad():
                    L = torch.mean(residual_fn(y)**2).item()
                    if L < 1e-16:
                        print('L too small; terminate.')
                        break

                # Jacobian
                J = self.jacobian(forward_fn, diff_parameters, create_graph=False)
                J = J.view(-1, J.shape[-1])
                JtJ = J.T @ J
                N = JtJ.shape[0]

                # Regularization matrix
                if self.regularization == 'I':
                    R = torch.eye(N, device=JtJ.device)
                elif self.regularization == 'diag':
                    R = torch.diag(torch.diag(JtJ).abs())
                else:
                    R = torch.diag(self.regularization)

                # Compute b = J^T @ residual
                bb = [
                    torch.autograd.grad(outputs=y, inputs=x, grad_outputs=residual_fn(y), retain_graph=True)[0]
                    for x in diff_parameters
                ]
                for i, bx in enumerate(bb):
                    if len(bx.shape) == 0:
                        bb[i] = torch.Tensor([bx.item()]).to(y.device)
                    if len(bx.shape) > 1:
                        bb[i] = torch.Tensor(bx.cpu().detach().numpy().flatten()).to(y.device)
                b = torch.cat(bb, axis=-1)
                del J, bb, y

                # Damping loop
                L_current = L + 1.0
                it_inner = 0
                while L_current >= L:
                    it_inner += 1
                    if it_inner > 20:
                        print('inner loop too many; Exiting damping loop.')
                        break

                    A = JtJ + lamb * R
                    try:
                        x_delta = torch.linalg.solve(A, b)
                    except torch._C._LinAlgError:
                        x_delta = torch.linalg.lstsq(A, b).solution
                    if torch.isnan(x_delta).sum():
                        print('x_delta NaN; Exiting damping loop')
                        break
                    x_delta_s = torch.split(x_delta, Ns)

                    x_delta_s = [*x_delta_s]
                    for xi in range(len(x_delta_s)):
                        x_delta_s[xi] = torch.reshape(x_delta_s[xi], NS[xi])

                    diff_parameters = change_parameters(scene, param_names, x_delta_s, sign=True)

                    with torch.no_grad():
                        L_current = torch.mean(residual_fn(forward_fn())**2).item()

                    del A

                    if L_current < L:
                        lamb /= self.mu
                        del x_delta_s
                        break

                    lamb *= 2.0 * self.mu
                    diff_parameters = change_parameters(scene, param_names, x_delta_s, sign=False)

                    if lamb > 1e16:
                        print('lambda too big; Exiting damping loop.')
                        del x_delta_s
                        break

                del JtJ, R, b

                x_increment = torch.mean(torch.abs(x_delta)).item()
                print('iter = {}: loss = {:.4e}, |x_delta| = {:.4e}'.format(
                    it, L, x_increment
                ))
                ls.append(L)
                if it > 0:
                    dls = np.abs(ls[-2] - L)
                    if dls < 1e-8:
                        print("|\u0394 loss| = {:.4e} < 1e-8; Exiting LM loop.".format(dls))
                        break

                if x_increment < 1e-8:
                    print("|x_delta| = {:.4e} < 1e-8; Exiting LM loop.".format(x_increment))
                    break

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_end = time.time()
        print('Elapsed time = {:e} seconds'.format(time_end - time_start))

        # Print final parameters
        for name in param_names:
            val = get_nested_attr(scene, name).cpu().detach().numpy()
            print('scene.{} = {}'.format(name, val))

        return ls
