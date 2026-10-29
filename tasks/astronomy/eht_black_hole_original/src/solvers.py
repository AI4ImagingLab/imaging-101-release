"""
Inverse Problem Solvers for Closure-Only VLBI Imaging
======================================================

Solvers
-------
ClosureRMLSolver
    RML imaging using closure phase and/or log closure amplitude chi-squared.
    Matches ehtim's Imager objective and optimization strategy:
    - Log-image transform (optimize in log(I) space)
    - Objective: Σ α_d*(χ²_d - 1) + Σ α_r*(-S_r(I))
    - Regularizers are negated to act as penalties (ehtim convention)

VisibilityRMLSolver
    Traditional visibility-based RML (comparison baseline).

Regularizers (matching ehtim's raw functions)
---------------------------------------------
GullSkillingRegularizer   Gull-Skilling entropy: Σ(I - P - I*log(I/P))  [non-positive]
SimpleEntropyRegularizer  Simple entropy: -Σ I*log(I/P)  [non-positive]
TVRegularizer             Total Variation (Huber-smoothed)  [non-negative]

Note: ehtim's `regularizer()` function negates entropy values before adding
to the objective. The solver handles this sign flip, not the regularizer classes.

Reference
---------
Chael et al. (2018). ApJ 857, 23.
ehtim: https://github.com/achael/eht-imaging
"""

import numpy as np
from scipy.optimize import minimize


# ═══════════════════════════════════════════════════════════════════════════
# Regularizers (matching ehtim's raw functions, norm_reg=False)
# ═══════════════════════════════════════════════════════════════════════════

class GullSkillingRegularizer:
    """
    Gull-Skilling entropy (matches ehtim's sgs, norm_reg=False).

    S(I) = Σ (I - P - I*log(I/P))   [non-positive, zero when I == P]

    Parameters
    ----------
    prior : (N, N) ndarray or None — prior image P
    """

    def __init__(self, prior=None):
        self.prior = prior

    def value_and_grad(self, x: np.ndarray):
        if self.prior is None:
            total = x.sum() + 1e-30
            prior = np.full_like(x, total / x.size)
        else:
            prior = self.prior

        imvec = x.ravel()
        priorvec = prior.ravel()

        val = float(np.sum(imvec - priorvec - imvec * np.log(imvec / priorvec)))
        grad = -np.log(imvec / priorvec)

        return val, grad.reshape(x.shape)


class SimpleEntropyRegularizer:
    """
    Simple entropy (matches ehtim's ssimple, norm_reg=False).

    S(I) = -Σ I*log(I/P)   [non-positive, zero when I == P]

    Parameters
    ----------
    prior : (N, N) ndarray or None — prior image P
    """

    def __init__(self, prior=None):
        self.prior = prior

    def value_and_grad(self, x: np.ndarray):
        if self.prior is None:
            total = x.sum() + 1e-30
            prior = np.full_like(x, total / x.size)
        else:
            prior = self.prior

        imvec = x.ravel()
        priorvec = prior.ravel()

        val = float(-np.sum(imvec * np.log(imvec / priorvec)))
        grad = -np.log(imvec / priorvec) - 1

        return val, grad.reshape(x.shape)


class TVRegularizer:
    """
    Total Variation regularizer (Huber-smoothed, periodic boundary).

    TV(x) = Σ √(dx² + dy² + ε²) - ε   [non-negative]

    ehtim's stv() returns -TV (negated). The solver handles the sign.

    Parameters
    ----------
    epsilon : float — Huber smoothing parameter
    """

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def value_and_grad(self, x: np.ndarray):
        eps = self.epsilon
        dx = np.roll(x, -1, axis=1) - x
        dy = np.roll(x, -1, axis=0) - x
        mag = np.sqrt(dx**2 + dy**2 + eps**2)
        val = float(np.sum(mag - eps))
        dv_dx = dx / mag - np.roll(dx / mag, 1, axis=1)
        dv_dy = dy / mag - np.roll(dy / mag, 1, axis=0)
        grad = dv_dx + dv_dy
        return val, grad


# ═══════════════════════════════════════════════════════════════════════════
# Solvers
# ═══════════════════════════════════════════════════════════════════════════

class ClosureRMLSolver:
    """
    RML Solver using closure quantity chi-squared.

    Matches ehtim's Imager class:
    - Optimizes in log-image space (transform=['log']), ensuring positivity
    - Objective: Σ α_d*(χ²_d - 1) + Σ α_r*(-S_r(I))
    - Entropy regularizers are negated to become penalties
    - TV regularizer is already non-negative (no negation needed)
    - Gradient includes chain rule for log transform: grad_log = I * grad_I

    Parameters
    ----------
    data_terms : dict
        Keys: 'cphase', 'logcamp', 'vis'. Values: weight alpha.
    reg_terms : dict
        Keys: 'gs', 'simple', 'tv'. Values: weight alpha.
    prior : (N, N) ndarray — Gaussian prior image
    n_iter : int — max L-BFGS-B iterations per round
    n_rounds : int — number of imaging rounds (like ehtim niter)
    """

    def __init__(self, data_terms=None, reg_terms=None, prior=None,
                 n_iter=300, n_rounds=3):
        self.data_terms = data_terms or {}
        self.reg_terms = reg_terms or {}
        self.prior = prior
        self.n_iter = n_iter
        self.n_rounds = n_rounds

    def reconstruct(self, model, obs_data, x0=None):
        """
        Run closure-based RML imaging.

        Parameters
        ----------
        model : ClosureForwardModel
        obs_data : dict with keys depending on data_terms:
            'vis_obs', 'sigma_vis' — for vis data term
            'cp_values_deg', 'cp_sigmas_deg', 'cp_u1/u2/u3' — for cphase
            'lca_values', 'lca_sigmas', 'lca_u1/u2/u3/u4' — for logcamp
        x0 : (N, N) initial image, defaults to prior

        Returns
        -------
        image : (N, N) reconstructed image
        """
        from src.physics_model import _ftmatrix

        N = model.N
        psize = model.pixel_size_rad

        if x0 is None:
            x0 = self.prior.copy() if self.prior is not None else np.ones((N, N)) / (N * N)

        # Pre-build A matrices for closure quantities
        A_cp = None
        if 'cphase' in self.data_terms:
            A_cp = (
                _ftmatrix(psize, N, obs_data['cp_u1']),
                _ftmatrix(psize, N, obs_data['cp_u2']),
                _ftmatrix(psize, N, obs_data['cp_u3']),
            )

        A_lca = None
        if 'logcamp' in self.data_terms:
            A_lca = (
                _ftmatrix(psize, N, obs_data['lca_u1']),
                _ftmatrix(psize, N, obs_data['lca_u2']),
                _ftmatrix(psize, N, obs_data['lca_u3']),
                _ftmatrix(psize, N, obs_data['lca_u4']),
            )

        # Build regularizers with sign convention matching ehtim's regularizer():
        # ehtim negates entropy (sgs, ssimple) so they become penalties (non-negative)
        # ehtim negates TV (stv returns -TV) — but our TVRegularizer returns +TV,
        # so we treat it the same as entropy: negate to match ehtim sign convention
        regularizers = []
        for reg_name, alpha in self.reg_terms.items():
            if reg_name == 'gs':
                regularizers.append((alpha, GullSkillingRegularizer(prior=self.prior)))
            elif reg_name == 'simple':
                regularizers.append((alpha, SimpleEntropyRegularizer(prior=self.prior)))
            elif reg_name == 'tv':
                regularizers.append((alpha, TVRegularizer()))

        DEGREE = np.pi / 180.0

        def objective_and_grad(logimvec):
            """Objective in log-image space (matching ehtim transform=['log'])."""
            imvec = np.exp(logimvec)  # change of variables: I = exp(log_I)
            img = imvec.reshape(N, N)
            total_obj = 0.0
            total_grad = np.zeros(N * N)

            # Closure phase data term
            if 'cphase' in self.data_terms and A_cp is not None:
                alpha = self.data_terms['cphase']
                cp_obs = obs_data['cp_values_deg'] * DEGREE
                cp_sig = obs_data['cp_sigmas_deg'] * DEGREE

                i1 = A_cp[0] @ imvec
                i2 = A_cp[1] @ imvec
                i3 = A_cp[2] @ imvec
                cp_model = np.angle(i1 * i2 * i3)

                n_cp = len(cp_obs)
                chisq = (2.0 / n_cp) * np.sum(
                    (1.0 - np.cos(cp_obs - cp_model)) / cp_sig**2
                )

                pref = np.sin(cp_obs - cp_model) / cp_sig**2
                g = (pref / i1) @ A_cp[0] + (pref / i2) @ A_cp[1] + (pref / i3) @ A_cp[2]
                grad_chisq = (-2.0 / n_cp) * np.imag(g)

                total_obj += alpha * (chisq - 1.0)
                total_grad += alpha * grad_chisq

            # Log closure amplitude data term
            if 'logcamp' in self.data_terms and A_lca is not None:
                alpha = self.data_terms['logcamp']
                lca_obs = obs_data['lca_values']
                lca_sig = obs_data['lca_sigmas']

                i1 = A_lca[0] @ imvec
                i2 = A_lca[1] @ imvec
                i3 = A_lca[2] @ imvec
                i4 = A_lca[3] @ imvec

                lca_model = (np.log(np.abs(i1)) + np.log(np.abs(i2))
                            - np.log(np.abs(i3)) - np.log(np.abs(i4)))

                n_ca = len(lca_obs)
                chisq = np.sum(np.abs((lca_obs - lca_model) / lca_sig)**2) / n_ca

                pp = (lca_obs - lca_model) / lca_sig**2
                g = ((pp / i1) @ A_lca[0] + (pp / i2) @ A_lca[1]
                   + (-pp / i3) @ A_lca[2] + (-pp / i4) @ A_lca[3])
                grad_chisq = (-2.0 / n_ca) * np.real(g)

                total_obj += alpha * (chisq - 1.0)
                total_grad += alpha * grad_chisq

            # Visibility data term (ehtim uses 2*n_vis normalization for complex vis)
            if 'vis' in self.data_terms:
                alpha = self.data_terms['vis']
                vis_obs = obs_data['vis_obs']
                sigma = obs_data['sigma_vis']

                vis_model = model.A @ imvec
                residual = vis_model - vis_obs
                n_vis = len(vis_obs)
                chisq = np.sum(np.abs(residual / sigma)**2) / (2 * n_vis)
                grad_chisq = (1.0 / n_vis) * (model.A.conj().T @ (residual / sigma**2)).real

                total_obj += alpha * (chisq - 1.0)
                total_grad += alpha * grad_chisq

            # Regularization — negate entropy regularizers (matching ehtim convention)
            # ehtim's regularizer() returns -sgs(), -ssimple(), -stv() so they are
            # non-negative penalties. Our classes return the raw (non-positive) values,
            # so we negate them here.
            for weight, reg in regularizers:
                rv, rg = reg.value_and_grad(img)
                total_obj += weight * (-rv)    # negate: -S is non-negative penalty
                total_grad += weight * (-rg.ravel())

            # Chain rule for log transform: d/d(log I) = I * d/dI
            total_grad *= imvec

            return float(total_obj), total_grad.astype(np.float64)

        # Multi-round optimization in log-image space (matching ehtim's niter)
        # ehtim: xinit = log(init_image), no bounds (log space is unconstrained)
        current = np.log(np.maximum(x0.ravel(), 1e-30)).astype(np.float64)

        for round_idx in range(self.n_rounds):
            result = minimize(
                objective_and_grad,
                current,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': self.n_iter, 'ftol': 1e-14, 'gtol': 1e-10},
            )
            current = result.x.copy()

        # Convert back from log space
        return np.exp(current).reshape(N, N)


class VisibilityRMLSolver:
    """
    Traditional visibility-based RML solver (comparison baseline).

    Minimizes: (1/2M) Σ |A*x - y|²/σ² + Σ λ_r R_r(x)

    Uses bounded L-BFGS-B (no log transform) with positivity constraint.
    """

    def __init__(self, regularizers=None, n_iter=500, positivity=True):
        self.regularizers = regularizers or []
        self.n_iter = n_iter
        self.positivity = positivity

    def reconstruct(self, model, vis, noise_std=1.0, x0=None):
        N = model.N
        if x0 is None:
            x0 = model.dirty_image(vis).clip(0)

        sigma = noise_std if np.isscalar(noise_std) else noise_std

        def objective_and_grad(x_flat):
            x = x_flat.reshape(N, N)
            vis_model = model.A @ x_flat
            residual = vis_model - vis

            if np.isscalar(sigma):
                chi2 = 0.5 * np.sum(np.abs(residual)**2) / sigma**2
                grad_chi2 = (model.A.conj().T @ residual).real / sigma**2
            else:
                chi2 = 0.5 * np.sum(np.abs(residual / sigma)**2)
                grad_chi2 = (model.A.conj().T @ (residual / sigma**2)).real

            reg_val = 0.0
            reg_grad = np.zeros(N * N)
            for weight, reg in self.regularizers:
                rv, rg = reg.value_and_grad(x)
                reg_val += weight * rv
                reg_grad += weight * rg.ravel()

            return chi2 + reg_val, grad_chi2 + reg_grad

        bounds = [(0.0, None)] * (N * N) if self.positivity else None
        result = minimize(
            objective_and_grad,
            x0.ravel().astype(np.float64),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': self.n_iter, 'ftol': 1e-14, 'gtol': 1e-9},
        )
        return result.x.reshape(N, N)
