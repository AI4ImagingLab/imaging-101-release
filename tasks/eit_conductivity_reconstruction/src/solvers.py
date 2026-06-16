# coding: utf-8
"""
EIT reconstruction solvers: BP, JAC (dynamic & static), and GREIT.

Adapted from pyEIT (https://github.com/liubenyuan/pyEIT).
Copyright (c) Benyuan Liu. All Rights Reserved.
Distributed under the (new) BSD License.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg as la

from .physics_model import EITForwardModel, sim2pts


# ---------------------------------------------------------------------------
# Helper utilities (adapted from pyeit/eit/interp2d.py)
# ---------------------------------------------------------------------------

class TriangleRasterizer:
    """Test whether points lie inside a set of triangles using barycentric areas."""

    def __init__(self, pts: np.ndarray, tri: np.ndarray) -> None:
        tp = pts[:, np.newaxis][tri].squeeze()
        tri_vec = tp[:, [1, 2, 0]] - tp
        self.tp = tp
        self.atot = np.abs(self._tri_area(tri_vec[:, 0], tri_vec[:, 1]))

    @staticmethod
    def _tri_area(bar0: np.ndarray, bar1: np.ndarray) -> np.ndarray:
        return bar0[:, 0] * bar1[:, 1] - bar0[:, 1] * bar1[:, 0]

    def _point_in_triangle(self, v: np.ndarray) -> bool:
        tv = self.tp - v
        a0 = self._tri_area(tv[:, 0], tv[:, 1])
        a1 = self._tri_area(tv[:, 1], tv[:, 2])
        a2 = self._tri_area(tv[:, 2], tv[:, 0])
        asum = np.sum(np.abs(np.vstack([a0, a1, a2])), axis=0)
        # add a margin for in-triangle test
        return np.any(asum <= 1.01 * self.atot)

    def points_in_triangles(self, varray: np.ndarray) -> np.ndarray:
        return np.array([self._point_in_triangle(v) for v in varray])


def _build_grid(
    pts: np.ndarray,
    n: int = 32,
    ext_ratio: float = 0.0,
    gc: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a regular mesh grid from a triangle point cloud.

    Parameters
    ----------
    pts : np.ndarray
        Nx2 array of (x, y) node positions.
    n : int
        Number of grid points per dimension.
    ext_ratio : float
        Extend the boundary by this fraction of the span.
    gc : bool
        If True, offset grids by half a step (grid correction).

    Returns
    -------
    xg, yg : np.ndarray
        Mesh grids.
    """
    x, y = pts[:, 0], pts[:, 1]
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_ext = (x_max - x_min) * ext_ratio
    y_ext = (y_max - y_min) * ext_ratio
    xv, xv_step = np.linspace(
        x_min - x_ext, x_max + x_ext, num=n, endpoint=False, retstep=True
    )
    yv, yv_step = np.linspace(
        y_min - y_ext, y_max + y_ext, num=n, endpoint=False, retstep=True
    )
    if gc:
        xv = xv + xv_step / 2.0
        yv = yv + yv_step / 2.0
    xg, yg = np.meshgrid(xv, yv, sparse=False, indexing="xy")
    return xg, yg


def rasterize(
    pts: np.ndarray,
    tri: np.ndarray,
    n: int = 32,
    ext_ratio: float = 0.0,
    gc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize a triangle mesh onto a regular grid (barycentric / "cg" method only).

    Parameters
    ----------
    pts : np.ndarray
        Nx2 node coordinates.
    tri : np.ndarray
        Mx3 element connectivity.
    n : int
        Grid resolution per dimension.
    ext_ratio : float
        Boundary extension ratio.
    gc : bool
        Grid correction flag.

    Returns
    -------
    xg, yg : np.ndarray
        Mesh grids.
    mask : np.ndarray
        Boolean array; True for grid points *outside* the mesh.
    """
    xg, yg = _build_grid(pts, n=n, ext_ratio=ext_ratio, gc=gc)
    points = np.vstack((xg.flatten(), yg.flatten())).T
    tr = TriangleRasterizer(pts[:, :2], tri)
    mask = ~tr.points_in_triangles(points)
    return xg, yg, mask


def _distance_matrix2d(xy: np.ndarray, xyi: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distance matrix between two 2-D point sets.

    Parameters
    ----------
    xy : np.ndarray
        Nx2 array.
    xyi : np.ndarray
        Mx2 array.

    Returns
    -------
    np.ndarray
        NxM distance matrix.
    """
    d0 = np.subtract.outer(xy[:, 0], xyi[:, 0])
    d1 = np.subtract.outer(xy[:, 1], xyi[:, 1])
    return np.hypot(d0, d1)


def weight_sigmod(
    xy: np.ndarray,
    xyi: np.ndarray,
    ratio: float = 0.05,
    s: float = 20.0,
) -> np.ndarray:
    """
    Sigmoid weighting matrix used by GREIT for element-to-grid interpolation.

    Parameters
    ----------
    xy : np.ndarray
        (x, y) element centres.
    xyi : np.ndarray
        (xi, yi) interpolated grid locations.
    ratio : float
        Desired radius as a fraction of max pairwise distance.
    s : float
        Controls the sigmoid decay rate.

    Returns
    -------
    w_mat : np.ndarray
        Column-normalised weighting matrix (n_elems x n_grid_pts).
    """
    d_mat = _distance_matrix2d(xy, xyi)
    d_max = np.max(d_mat)
    d_mat = 5.0 * d_mat / d_max
    r0 = 5.0 * ratio
    weight = 1.0 / (1 + np.exp(s * (d_mat - r0)))
    return weight / weight.sum(axis=0)


# ---------------------------------------------------------------------------
# Shared JAC helper
# ---------------------------------------------------------------------------

def _compute_h_jac(
    jac: np.ndarray,
    p: float,
    lamb: float,
    method: str = "kotre",
) -> np.ndarray:
    """
    Compute the pseudo-inverse matrix H for Jacobian-based solvers.

        H = (J^T J + lamb * R)^{-1} J^T

    Parameters
    ----------
    jac : np.ndarray
        Jacobian matrix (n_meas x n_elems).
    p : float
        Regularisation exponent (only used for ``method='kotre'``).
    lamb : float
        Regularisation weight.
    method : str
        ``'kotre'``, ``'lm'`` (Levenberg-Marquardt / NOSER), or ``'dgn'``
        (damped Gauss-Newton).

    Returns
    -------
    np.ndarray
        H matrix.
    """
    j_w_j = np.dot(jac.T, jac)
    if method == "kotre":
        r_mat = np.diag(np.diag(j_w_j) ** p)
    elif method == "lm":
        r_mat = np.diag(np.diag(j_w_j))
    else:
        # Damped Gauss-Newton
        r_mat = np.eye(jac.shape[1])
    return np.dot(la.inv(j_w_j + lamb * r_mat), jac.T)


# ---------------------------------------------------------------------------
# Back-Projection
# ---------------------------------------------------------------------------

class BPReconstructor:
    """
    Naive (Euclidean) back-projection solver.

    Parameters
    ----------
    weight : str
        Weighting scheme.  ``'none'`` (default) or ``'simple'`` (radial).
    """

    def __init__(self, weight: str = "none") -> None:
        self.weight = weight

    def reconstruct(
        self,
        model: EITForwardModel,
        v1: np.ndarray,
        v0: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Reconstruct conductivity changes via back-projection.

        Parameters
        ----------
        model : EITForwardModel
            Forward model providing mesh and ``compute_b_matrix``.
        v1 : np.ndarray
            Current measurement frame.
        v0 : np.ndarray
            Reference measurement frame.
        normalize : bool
            If True, use BP-specific sign normalisation.

        Returns
        -------
        ds : np.ndarray
            Estimated conductivity changes on mesh nodes.
        """
        B = model.compute_b_matrix()

        if self.weight == "simple":
            # weight by distance from centre
            node = model.mesh.node
            d = np.sqrt(np.sum(node ** 2, axis=1))
            r = np.max(d)
            w = (1.01 * r - d) / (1.01 * r)
            num_voltages = B.shape[0]
            weights = np.dot(np.ones((num_voltages, 1)), w.reshape(1, -1))
            B = weights * B

        H = B.T

        if normalize:
            dv = (v1 - v0) / np.sign(v0.real)
        else:
            dv = v1 - v0

        ds = -np.dot(H, dv)
        return ds


# ---------------------------------------------------------------------------
# GREIT
# ---------------------------------------------------------------------------

class GREITReconstructor:
    """
    GREIT (Graz consensus Reconstruction algorithm for EIT) solver.

    Parameters
    ----------
    p : float
        Noise covariance exponent for regularisation.
    lamb : float
        Regularisation parameter.
    n : int
        Grid resolution per dimension.
    s : float
        Sigmoid decay parameter for grid weighting.
    ratio : float
        Desired radius ratio for sigmoid weighting.
    jac_normalized : bool
        Whether to normalise the Jacobian by forward voltages.

    References
    ----------
    Adler, A. et al. "GREIT: a unified approach to 2D linear EIT
    reconstruction of lung images." Physiological Measurement 30.6 (2009): S35.
    """

    def __init__(
        self,
        p: float = 0.50,
        lamb: float = 0.01,
        n: int = 32,
        s: float = 20.0,
        ratio: float = 0.1,
        jac_normalized: bool = True,
    ) -> None:
        self.p = p
        self.lamb = lamb
        self.n = n
        self.s = s
        self.ratio = ratio
        self.jac_normalized = jac_normalized

    def reconstruct(
        self,
        model: EITForwardModel,
        v1: np.ndarray,
        v0: np.ndarray,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct conductivity changes on a regular pixel grid.

        Parameters
        ----------
        model : EITForwardModel
            Forward model providing mesh, Jacobian, etc.
        v1 : np.ndarray
            Current measurement frame.
        v0 : np.ndarray
            Reference measurement frame.
        normalize : bool
            If True, normalise ``dv`` by ``|v0|``.

        Returns
        -------
        xg, yg : np.ndarray
            Grid coordinates.
        ds : np.ndarray
            Reconstructed conductivity change image (n x n), with values
            outside the mesh set to zero.
        """
        # 1. Build grids and mask
        pts = model.mesh.node
        tri = model.mesh.element
        xg, yg, mask = rasterize(pts, tri, n=self.n)

        # 2. Compute grid weights using sigmoid interpolation
        xy = model.mesh.elem_centers
        xyi = np.vstack((xg.flatten(), yg.flatten())).T
        w_mat = weight_sigmod(xy, xyi, ratio=self.ratio, s=self.s)

        # 3. Compute Jacobian
        jac, _v0_fwd = model.compute_jac(normalize=self.jac_normalized)

        # 4. Compute H = w_mat.T @ J.T @ inv(J @ J.T + lamb * R)
        j_j_w = np.dot(jac, jac.T)
        r_mat = np.diag(np.diag(j_j_w) ** self.p)
        jac_inv = la.inv(j_j_w + self.lamb * r_mat)
        H = np.dot(np.dot(w_mat.T, jac.T), jac_inv)

        # 5. Normalise dv
        if normalize:
            dv = (v1 - v0) / np.abs(v0)
        else:
            dv = v1 - v0

        # 6. Reconstruct
        ds = -np.dot(H, dv)

        # 7. Mask and reshape
        ds[mask] = 0.0
        ds = ds.reshape(xg.shape)
        return xg, yg, ds


# ---------------------------------------------------------------------------
# JAC Dynamic
# ---------------------------------------------------------------------------

class JACDynamicReconstructor:
    """
    Jacobian-based dynamic (time-difference) EIT solver.

    Solves for conductivity *changes* between two measurement frames:
        ds = -H (v1 - v0)

    Parameters
    ----------
    p : float
        Regularisation exponent (Kotre).
    lamb : float
        Regularisation weight.
    method : str
        ``'kotre'``, ``'lm'``, or ``'dgn'``.
    jac_normalized : bool
        Normalise Jacobian by forward voltages.
    """

    def __init__(
        self,
        p: float = 0.5,
        lamb: float = 0.01,
        method: str = "kotre",
        jac_normalized: bool = True,
    ) -> None:
        self.p = p
        self.lamb = lamb
        self.method = method
        self.jac_normalized = jac_normalized

    def reconstruct(
        self,
        model: EITForwardModel,
        v1: np.ndarray,
        v0: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Reconstruct element-wise conductivity changes.

        Parameters
        ----------
        model : EITForwardModel
            Forward model.
        v1 : np.ndarray
            Current measurement frame.
        v0 : np.ndarray
            Reference measurement frame.
        normalize : bool
            If True, normalise ``dv`` by ``|v0|``.

        Returns
        -------
        ds : np.ndarray
            Element-wise conductivity changes.
        """
        # 1. Compute Jacobian
        jac, _v0_fwd = model.compute_jac(normalize=self.jac_normalized)

        # 2. Compute H
        H = _compute_h_jac(jac, self.p, self.lamb, self.method)

        # 3. Normalise dv
        if normalize:
            dv = (v1 - v0) / np.abs(v0)
        else:
            dv = v1 - v0

        # 4. Reconstruct
        ds = -np.dot(H, dv)
        return ds


