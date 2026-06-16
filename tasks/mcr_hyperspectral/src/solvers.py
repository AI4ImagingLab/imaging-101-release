"""MCR-AR solvers with various regressors and constraints.

Self-contained implementation (no pyMCR or lmfit dependency) of five
MCR variants from the pyMCR paper:
  1. MCR-ALS  — OLS regressors, non-negativity + normalization
  2. MCR-NNLS — NNLS regressors, normalization only on C
  3. MCR-AR Gauss — NNLS + Gaussian shape constraint on spectra
  4. MCR-AR Ridge — Ridge regression on S^T, OLS on C
  5. MCR-AR Lasso — Lasso regression on C, OLS on S^T
"""

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import nnls, curve_fit
from sklearn.linear_model import Ridge, Lasso


# ============================================================
# Regressors  (replace pymcr.regressors)
# ============================================================

class _OLS:
    """Ordinary least squares: solve AX = B for X."""

    def fit(self, A, B):
        self.X_, _, _, _ = lstsq(A, B)

    @property
    def coef_(self):
        return self.X_.T


class _NNLS:
    """Non-negative least squares: solve AX = B for X >= 0."""

    def fit(self, A, B):
        if B.ndim == 2:
            N = B.shape[1]
            self.X_ = np.zeros((A.shape[1], N))
            for col in range(N):
                self.X_[:, col], _ = nnls(A, B[:, col])
        else:
            self.X_, _ = nnls(A, B)

    @property
    def coef_(self):
        return self.X_.T


# ============================================================
# Constraints  (replace pymcr.constraints)
# ============================================================

class ConstraintNonneg:
    """Set all negative values to zero."""

    def transform(self, A):
        A *= (A > 0)
        return A


class ConstraintNorm:
    """Normalise so each row (axis=-1) or column (axis=0) sums to 1."""

    def __init__(self, axis=-1):
        self.axis = axis

    def transform(self, A):
        if self.axis in (-1, 1):
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            A /= row_sums
        else:
            col_sums = A.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            A /= col_sums
        return A


class ConstraintSingleGauss:
    """Enforce a Gaussian shape on each spectral component via NLLS fitting.

    Uses scipy.optimize.curve_fit to fit each row (or column) to a
    Gaussian + constant model, replacing lmfit.

    Parameters
    ----------
    alpha : float
        Blending weight in [0, 1]. alpha=1 uses pure Gaussian fit;
        alpha=0 returns the original data unchanged.
    copy : bool
        If True, return blended copy; otherwise modify in-place.
    axis : int
        Axis along which to perform the Gaussian fitting.
    """

    def __init__(self, alpha=1.0, copy=False, axis=-1):
        self.copy = copy
        self.axis = axis
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")
        self.alpha = alpha

    @staticmethod
    def _gauss_plus_const(x, amplitude, center, sigma, const):
        """Gaussian + constant baseline model."""
        return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) + const

    def transform(self, A):
        """Apply Gaussian fitting constraint to each row/column."""
        n_components = list(A.shape)
        x = np.arange(n_components[self.axis], dtype=float)
        n_components.pop(self.axis)
        assert len(n_components) == 1, "Input must be 2D"
        n_components = n_components[0]

        A_fit = np.zeros_like(A)

        for num in range(n_components):
            if self.axis in (-1, 1):
                y = A[num, :]
            else:
                y = A[:, num]

            # Initial guesses
            peak_idx = np.argmax(y)
            amp_guess = float(y[peak_idx] - y.min())
            center_guess = float(peak_idx)
            sigma_guess = float(len(x) / 6)
            const_guess = float(y.min())

            try:
                popt, _ = curve_fit(
                    self._gauss_plus_const, x, y,
                    p0=[amp_guess, center_guess, sigma_guess, const_guess],
                    bounds=([0, x.min(), 0, -np.inf],
                            [np.inf, x.max(), np.inf, np.inf]),
                    maxfev=5000,
                )
                best_fit = self._gauss_plus_const(x, *popt)
            except RuntimeError:
                best_fit = y.copy()

            if self.axis in (-1, 1):
                A_fit[num, :] = best_fit
            else:
                A_fit[:, num] = best_fit

        if self.copy:
            return self.alpha * A_fit + (1 - self.alpha) * A
        else:
            temp = A.copy()
            A *= 0
            A += self.alpha * A_fit + (1 - self.alpha) * temp
            return A


# ============================================================
# MCR-AR solver  (replace pymcr.mcr.McrAR)
# ============================================================

class McrAR:
    """Multivariate Curve Resolution — Alternating Regression.

    Faithfully reproduces the pyMCR algorithm: alternating regression
    with per-half-iteration error tracking, constraint application,
    and multiple convergence criteria.

    Parameters
    ----------
    max_iter : int
        Maximum number of full iterations.
    tol_increase : float or None
        Maximum allowed factor increase in error per half-iteration.
    tol_n_increase : int or None
        Maximum consecutive half-iteration error increases.
    tol_n_above_min : int or None
        Maximum half-iterations above error minimum.
    tol_err_change : float or None
        Minimum error change per full iteration to continue.
    c_regr, st_regr : str or sklearn-like object
        Regressor for C and S^T.  "OLS" / "NNLS" are converted to
        built-in implementations; any object with fit()/coef_ works.
    c_constraints, st_constraints : list
        Constraint objects applied after each regression step.
    """

    def __init__(self, max_iter=50, tol_increase=0.0, tol_n_increase=10,
                 tol_n_above_min=10, tol_err_change=None,
                 c_regr="OLS", st_regr="OLS",
                 c_constraints=None, st_constraints=None):
        self.max_iter = max_iter
        self.tol_increase = tol_increase
        self.tol_n_increase = tol_n_increase
        self.tol_n_above_min = tol_n_above_min
        self.tol_err_change = tol_err_change
        self.c_regressor = self._resolve_regr(c_regr)
        self.st_regressor = self._resolve_regr(st_regr)
        self.c_constraints = c_constraints or []
        self.st_constraints = st_constraints or []

        # State (populated by fit)
        self.C_ = None
        self.ST_ = None
        self.C_opt_ = None
        self.ST_opt_ = None
        self.n_iter = 0
        self.n_iter_opt = 0
        self.err = []

    @staticmethod
    def _resolve_regr(regr):
        if isinstance(regr, str):
            if regr.upper() == "OLS":
                return _OLS()
            elif regr.upper() == "NNLS":
                return _NNLS()
            else:
                raise ValueError("Unknown regressor string: {}".format(regr))
        return regr

    @staticmethod
    def _mse(D, D_calc):
        return float(np.sum((D - D_calc) ** 2) / D.size)

    def _is_min_err(self, err_val):
        if len(self.err) == 0:
            return True
        return err_val <= min(self.err)

    def _regress_c(self, D):
        """Solve for C given current S^T.

        For OLS/NNLS: solve (ST^T) X = D^T => coef_ = X^T = C
        For sklearn:  fit(ST^T, D^T) => coef_ shape (n_pixels, n_components) = C
        """
        regr = self.c_regressor
        if isinstance(regr, (_OLS, _NNLS)):
            regr.fit(self.ST_.T, D.T)
            return regr.coef_.copy()
        else:
            # sklearn: fit(X, y) with X=(n_freq, k), y=(n_freq, n_pixels)
            # coef_ = (n_pixels, k) = C
            regr.fit(self.ST_.T, D.T)
            coef = regr.coef_
            if coef.ndim == 1:
                return coef.reshape(1, -1).copy()
            return coef.copy()

    def _regress_st(self, D):
        """Solve for S^T given current C.

        For OLS/NNLS: solve C X = D => coef_ = X^T, so S^T = coef_^T
        For sklearn:  fit(C, D) => coef_ shape (n_freq, k), so S^T = coef_^T
        """
        regr = self.st_regressor
        if isinstance(regr, (_OLS, _NNLS)):
            regr.fit(self.C_, D)
            return regr.coef_.T.copy()
        else:
            # sklearn: fit(X, y) with X=(n_pixels, k), y=(n_pixels, n_freq)
            # coef_ = (n_freq, k), so S^T = coef_.T = (k, n_freq)
            regr.fit(self.C_, D)
            return regr.coef_.T.copy()

    def fit(self, D, ST=None, verbose=True):
        """Run the MCR-AR alternating regression.

        Parameters
        ----------
        D : ndarray, shape (n_pixels, n_freq)
            Observed data matrix.
        ST : ndarray, shape (n_components, n_freq)
            Initial spectral guess.
        verbose : bool
            Print convergence messages.
        """
        D = np.asarray(D, dtype=float)
        self.ST_ = ST.copy()
        self.err = []
        self.n_iter = 0
        n_increase = 0
        n_above_min = 0

        for iteration in range(self.max_iter):
            self.n_iter = iteration + 1

            # --- Half-iteration 1: solve for C ---
            C_temp = self._regress_c(D)
            for constr in self.c_constraints:
                C_temp = constr.transform(C_temp)

            D_calc = C_temp @ self.ST_
            err_c = self._mse(D, D_calc)

            if self._is_min_err(err_c):
                self.C_opt_ = C_temp.copy()
                self.ST_opt_ = self.ST_.copy()
                self.n_iter_opt = self.n_iter
                n_above_min = 0
            else:
                n_above_min += 1

            if self.tol_n_above_min is not None and n_above_min > self.tol_n_above_min:
                if verbose:
                    print("N above min exceeded. Exiting.")
                break

            if len(self.err) == 0:
                self.err.append(err_c)
                self.C_ = C_temp.copy()
            elif self.tol_increase is None or err_c <= self.err[-1] * (1 + self.tol_increase):
                self.err.append(err_c)
                self.C_ = C_temp.copy()
            else:
                if verbose:
                    print("Tolerance increase exceeded. Exiting.")
                break

            if len(self.err) > 1:
                if self.err[-1] > self.err[-2]:
                    n_increase += 1
                else:
                    n_increase = 0

            if self.tol_n_increase is not None and n_increase > self.tol_n_increase:
                if verbose:
                    print("N consecutive increases exceeded. Exiting.")
                break

            # --- Half-iteration 2: solve for S^T ---
            ST_temp = self._regress_st(D)
            for constr in self.st_constraints:
                ST_temp = constr.transform(ST_temp)

            D_calc = self.C_ @ ST_temp
            err_st = self._mse(D, D_calc)

            if self._is_min_err(err_st):
                self.C_opt_ = self.C_.copy()
                self.ST_opt_ = ST_temp.copy()
                self.n_iter_opt = self.n_iter
                n_above_min = 0
            else:
                n_above_min += 1

            if self.tol_n_above_min is not None and n_above_min > self.tol_n_above_min:
                if verbose:
                    print("N above min exceeded. Exiting.")
                break

            if self.tol_increase is None or err_st <= self.err[-1] * (1 + self.tol_increase):
                self.err.append(err_st)
                self.ST_ = ST_temp.copy()
            else:
                if verbose:
                    print("Tolerance increase exceeded. Exiting.")
                break

            if len(self.err) > 1:
                if self.err[-1] > self.err[-2]:
                    n_increase += 1
                else:
                    n_increase = 0

            if self.tol_n_increase is not None and n_increase > self.tol_n_increase:
                if verbose:
                    print("N consecutive increases exceeded. Exiting.")
                break

            # --- End-of-iteration checks ---
            if self.n_iter >= self.max_iter:
                if verbose:
                    print("Max iterations reached ({}).".format(self.max_iter))
                break

            if self.tol_err_change is not None and len(self.err) > 2:
                err_differ = abs(self.err[-1] - self.err[-3])
                if err_differ < abs(self.tol_err_change):
                    if verbose:
                        print("Change in err below tol_err_change ({:.4e}). Exiting.".format(
                            err_differ))
                    break
        else:
            if verbose:
                print("Max iterations reached ({}).".format(self.max_iter))

    @property
    def D_opt_(self):
        return self.C_opt_ @ self.ST_opt_


# ============================================================
# Method configurations and runners
# ============================================================

def build_method_configs():
    """Return the five MCR method configurations.

    Returns
    -------
    configs : list of dict
        Each dict has keys: 'name', 'c_regr', 'st_regr',
        'c_constraints', 'st_constraints'.
    """
    return [
        {
            "name": "MCR-ALS",
            "c_regr": "OLS",
            "st_regr": "OLS",
            "c_constraints": [ConstraintNonneg(), ConstraintNorm()],
            "st_constraints": [ConstraintNonneg()],
        },
        {
            "name": "MCR-NNLS",
            "c_regr": "NNLS",
            "st_regr": "NNLS",
            "c_constraints": [ConstraintNorm()],
            "st_constraints": [],
        },
        {
            "name": "MCR-AR Gauss",
            "c_regr": "NNLS",
            "st_regr": "NNLS",
            "c_constraints": [ConstraintNonneg(), ConstraintNorm()],
            "st_constraints": [ConstraintNonneg(), ConstraintSingleGauss(alpha=1)],
        },
        {
            "name": "MCR-AR Ridge",
            "c_regr": "OLS",
            "st_regr": Ridge(alpha=10, fit_intercept=False, random_state=0),
            "c_constraints": [ConstraintNonneg(), ConstraintNorm()],
            "st_constraints": [ConstraintNonneg()],
        },
        {
            "name": "MCR-AR Lasso",
            "c_regr": Lasso(alpha=1, fit_intercept=True, positive=True, random_state=0),
            "st_regr": "OLS",
            "c_constraints": [ConstraintNonneg(), ConstraintNorm()],
            "st_constraints": [ConstraintNonneg()],
        },
    ]


# Solver parameters (not in meta_data.json per CLAUDE.md rules)
_MCR_PARAMS = {
    "max_iter": 200,
    "tol_increase": 1e10,
    "tol_n_increase": 10000,
    "tol_n_above_min": 10000,
    "tol_err_change": 1e-14,
}


def match_components(C_est, conc_ravel, n_components):
    """Find the permutation matching estimated to true components.

    Uses minimum squared error to assign each true component to the
    closest estimated component column.

    Parameters
    ----------
    C_est : ndarray, shape (n_pixels, n_components)
        Estimated concentration matrix.
    conc_ravel : ndarray, shape (n_pixels, n_components)
        True concentration matrix.
    n_components : int
        Number of components.

    Returns
    -------
    select : list of int
        Index mapping: select[k] is the estimated column matching true column k.
    """
    select = []
    for k in range(n_components):
        select.append(
            int(np.argmin(np.sum((C_est - conc_ravel[:, k : k + 1]) ** 2, axis=0)))
        )
    return select


def run_mcr(hsi_noisy, initial_spectra, config, mcr_params=None):
    """Run a single MCR-AR method.

    Parameters
    ----------
    hsi_noisy : ndarray, shape (n_pixels, n_freq)
        Noisy observation matrix.
    initial_spectra : ndarray, shape (n_components, n_freq)
        Initial spectral guess.
    config : dict
        Method configuration from build_method_configs().
    mcr_params : dict, optional
        Override default MCR solver parameters.

    Returns
    -------
    mcr_obj : McrAR
        Fitted MCR object with C_opt_, ST_opt_, D_opt_, err attributes.
    """
    params = dict(_MCR_PARAMS)
    if mcr_params is not None:
        params.update(mcr_params)

    mcr_obj = McrAR(
        **params,
        st_regr=config["st_regr"],
        c_regr=config["c_regr"],
        c_constraints=config["c_constraints"],
        st_constraints=config["st_constraints"],
    )
    mcr_obj.fit(hsi_noisy, ST=initial_spectra.copy(), verbose=False)
    return mcr_obj


def run_all_methods(hsi_noisy, initial_spectra, conc_ravel, spectra, mcr_params=None):
    """Run all five MCR methods and collect comparison metrics.

    Parameters
    ----------
    hsi_noisy : ndarray, shape (n_pixels, n_freq)
        Noisy observation matrix.
    initial_spectra : ndarray, shape (n_components, n_freq)
        Initial spectral guess.
    conc_ravel : ndarray, shape (n_pixels, n_components)
        True concentrations (for metric computation).
    spectra : ndarray, shape (n_components, n_freq)
        True spectra (for metric computation).
    mcr_params : dict, optional
        Override default MCR solver parameters.

    Returns
    -------
    results : list of dict
        Per-method results with keys: 'name', 'mcr', 'select',
        'mse', 'n_iter', 'n_iter_opt'.
    """
    from timeit import default_timer as timer

    configs = build_method_configs()
    n_components = spectra.shape[0]
    results = []

    for config in configs:
        print("-------- {} --------".format(config["name"]))
        t0 = timer()
        mcr_obj = run_mcr(hsi_noisy, initial_spectra, config, mcr_params)
        elapsed = timer() - t0
        print("Final MSE: {:.7e}  ({:.1f}s, {} iters)".format(
            mcr_obj.err[-1], elapsed, mcr_obj.n_iter))

        select = match_components(mcr_obj.C_opt_, conc_ravel, n_components)

        results.append({
            "name": config["name"],
            "mcr": mcr_obj,
            "select": select,
            "mse": float(np.min(mcr_obj.err)),
            "n_iter": mcr_obj.n_iter,
            "n_iter_opt": mcr_obj.n_iter_opt,
            "elapsed": elapsed,
        })

    return results
