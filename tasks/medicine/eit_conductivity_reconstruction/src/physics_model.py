# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
# pylint: disable=too-many-instance-attributes, too-many-arguments
"""
Physics model for EIT (Electrical Impedance Tomography) forward simulation.

This module contains code extracted and adapted from the pyEIT library
(https://github.com/liubenyuan/pyEIT) for use as a self-contained benchmark
task. It includes mesh data structures, EIT measurement protocols, FEM forward
solvers, and element-to-node interpolation utilities.

Original pyEIT code is Copyright (c) Benyuan Liu. All rights reserved.
Distributed under the (new) BSD License.
"""
from __future__ import absolute_import, division, print_function, annotations

from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

import numpy as np
import numpy.linalg as la
from scipy import sparse
import scipy.sparse.linalg
import scipy.linalg
from scipy.sparse import coo_matrix
import warnings


# ---------------------------------------------------------------------------
# Mesh data structures (adapted from pyeit/mesh/wrapper.py)
# ---------------------------------------------------------------------------


@dataclass
class PyEITMesh:
    """
    Pyeit buid-in mesh object

    Parameters
    ----------
    node : np.ndarray
        node of the mesh of shape (n_nodes, 2), (n_nodes, 3)
    element : np.ndarray
        elements of the mesh of shape (n_elem, 3) for 2D mesh, (n_elem, 4) for 3D mesh
    perm : Union[int, float, np.ndarray], optional
        permittivity on elements; shape (n_elems,), by default `None`.
        If `None`, a uniform permittivity on elements with a value 1 will be generated.
        If perm is int or float, uniform permittivity on elements with value of perm will be generated.
    el_pos : np.ndarray
        node corresponding to each electrodes of shape (n_el, 1)
    ref_node : int
        reference node. ref_node should not be on electrodes, default 0.
    """

    node: np.ndarray
    element: np.ndarray
    perm: Union[int, float, complex, np.ndarray] = field(default_factory=lambda: 1.0)
    el_pos: np.ndarray = field(default_factory=lambda: np.arange(16))
    ref_node: int = field(default_factory=lambda: 0)

    def __post_init__(self) -> None:
        """Checking of the inputs"""
        self.element = self._check_element(self.element)
        self.node = self._check_node(self.node)
        # self.perm = self.get_valid_perm_array(self.perm)
        self.ref_node = self._check_ref_node(self.ref_node)

    def _check_element(self, element: np.ndarray) -> np.ndarray:
        """
        Check nodes element
        return nodes [x,y,z]

        Parameters
        ----------
        element : np.ndarray
            nodes [x,y] ; shape (n_elem,3)
            nodes [x,y,z] ; shape (n_nodes,4)

        Returns
        -------
        np.ndarray
            elements ; shape (n_elems, 3) or (n_elems, 4)

        Raises
        ------
        TypeError
            raised if element is not ndarray of correct shape
        """
        if not isinstance(element, np.ndarray):
            raise TypeError(f"Wrong type of {element=}, expected an ndarray")
        if element.ndim != 2:
            raise TypeError(
                f"Wrong shape of {element.shape=}, expected an ndarray with 2 dimensions"
            )
        if element.shape[1] not in [3, 4]:
            raise TypeError(
                f"Wrong shape of {element.shape=}, expected an ndarray of shape (n_nodes,2) or (n_nodes,3)"
            )

        return element

    def _check_node(self, node: np.ndarray) -> np.ndarray:
        """
        Check nodes shape
        return nodes [x,y,z]

        Parameters
        ----------
        node : np.ndarray, optional
            nodes [x,y] ; shape (n_nodes,2) (in that case z will be set 0)
            nodes [x,y,z] ; shape (n_nodes,3)

        Returns
        -------
        np.ndarray
            nodes [x,y,z] ; shape (n_nodes,3)

        Raises
        ------
        TypeError
            raised if node is not ndarray of correct shape
        """
        if not isinstance(node, np.ndarray):
            raise TypeError(f"Wrong type of {node=}, expected an ndarray")
        if node.ndim != 2:
            raise TypeError(
                f"Wrong shape of {node.shape=}, expected an ndarray with 2 dimensions"
            )
        if node.shape[1] not in [2, 3]:
            raise TypeError(
                f"Wrong shape of {node.shape=}, expected an ndarray of shape (n_nodes,2) or (n_nodes,3)"
            )
        # convert nodes [x,y] to nodes [x,y,0]
        if node.shape[1] == 2:
            node = np.hstack((node, np.zeros((node.shape[0], 1))))

        return node

    def get_valid_perm_array(
        self, perm: Union[int, float, complex, np.ndarray] = 1.0
    ) -> np.ndarray:
        """
        Return a permittivity NDArray on element

        Parameters
        ----------
        perm : Union[int, float, complex, np.ndarray], optional
            Permittivity on elements ; shape (n_elems,), by default `None`.
            If `None`, a uniform permittivity on elements with a value 1 will be used.
            If perm is int or float, uniform permittivity on elements will be used.

        Returns
        -------
        np.ndarray
            permittivity on elements ; shape (n_elems,)

        Raises
        ------
        TypeError
            if perm is not a valid type or shape
        """

        if perm is None:
            return np.ones(self.n_elems, dtype=float)
        elif isinstance(perm, (int, float)):
            return np.ones(self.n_elems, dtype=float) * perm
        elif isinstance(perm, complex):
            return np.ones(self.n_elems, dtype=complex) * perm

        if not isinstance(perm, np.ndarray) or perm.shape != (self.n_elems,):
            raise TypeError(
                f"Wrong type/shape of {perm=}, expected an ndarray; shape ({self.n_elems}, )"
            )
        return perm

    def _check_ref_node(self, ref: int = 0) -> int:
        """
        Return a valid reference electrode node

        Parameters
        ----------
        ref : int, optional
            node number of reference node, by default 0
            If the choosen node is on electrode node, a node-list in
            np.arange(0, len(el_pos)+1) will be checked iteratively until
            a non-electrode node is selected.

        returns
        -------
        int
            valid reference electrode node
        """
        default_ref = np.setdiff1d(np.arange(len(self.el_pos) + 1), self.el_pos)[0]
        return ref if ref not in self.el_pos else int(default_ref)

    @property
    def perm_array(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            ndarray has a shape (n_elems,)
        """
        return self.get_valid_perm_array(self.perm)

    @property
    def n_nodes(self) -> int:
        """
        Returns
        -------
        int
            number of nodes contained in the mesh
        """
        return self.node.shape[0]

    @property
    def n_elems(self) -> int:
        """
        Returns
        -------
        int
            number of elements contained in the mesh
        """
        return self.element.shape[0]

    @property
    def n_vertices(self) -> int:
        """
        Returns
        -------
        int
            number of vertices of the elements contained in the mesh
        """
        return self.element.shape[1]

    @property
    def n_el(self) -> int:
        """
        Returns
        -------
        int
            number of electrodes
        """
        return self.el_pos.shape[0]

    @property
    def elem_centers(self):
        """
        Returns
        -------
        np.ndarray
            center of the nodes [x,y,z]; shape (n_elems,3)
        """
        return np.mean(self.node[self.element], axis=1)

    @property
    def dtype(self):
        """
        Returns
        -------
        Type
            data type of permmitivity
        """
        if isinstance(self.perm, (int, float)):
            return float
        elif isinstance(self.perm, complex):
            return complex
        elif isinstance(self.perm, np.ndarray):
            return self.perm.dtype

    @property
    def is_3D(self) -> bool:
        """
        Returns
        -------
        bool
            True if the mesh is a 3D mesh (use elements with 4 vertices)
        """
        return self.n_vertices == 4

    @property
    def is_2D(self) -> bool:
        """
        Returns
        -------
        bool
            True if the mesh is a 2D mesh (use elements with 3 vertices)
        """
        return self.n_vertices == 3


# ---------------------------------------------------------------------------
# Anomaly data structures (adapted from pyeit/mesh/wrapper.py)
# ---------------------------------------------------------------------------


@dataclass
class PyEITAnomaly_Circle:
    """
    Pyeit Anomaly for simulation purpose, 2D circle.

    Parameters
    ----------
    center : Union[np.ndarray, list]
        center of the anomaly
    r : float
        radius of the circle
    perm : float
        permittivity of the anomaly
    """

    center: Union[np.ndarray, list]
    r: float = 1.0
    perm: float = 1.0

    def __post_init__(self):
        if isinstance(self.center, list):
            self.center = np.array(self.center)

    def mask(self, pts: np.ndarray) -> np.ndarray:
        """
        Return mask corresponding to the pts contained in the circle anomaly.

        Parameters
        ----------
        pts : np.ndarray
            points to test; shape (n_pts, 2) or (n_pts, 3)

        Returns
        -------
        np.ndarray
            boolean mask; shape (n_pts,)
        """
        return (
            np.sqrt(np.sum((pts[:, :2] - self.center[:2]) ** 2, axis=1)) < self.r
        )


# ---------------------------------------------------------------------------
# set_perm (adapted from pyeit/mesh/wrapper.py)
# ---------------------------------------------------------------------------


def set_perm(
    mesh: PyEITMesh,
    anomaly: Union[PyEITAnomaly_Circle, List[PyEITAnomaly_Circle]],
    background: Optional[float] = None,
) -> PyEITMesh:
    """wrapper for pyEIT interface

    Note
    ----
    update permittivity of mesh, if specified.

    Parameters
    ----------
    mesh: PyEITMesh
        mesh object
    anomaly: Union[PyEITAnomaly_Circle, List[PyEITAnomaly_Circle]], optional
        anomaly object or list of anomaly objects contains,
        all permittivity on triangles whose distance to (x,y) are less than (d)
        will be replaced with a new value, 'perm' may be a complex value.
    background: float, optional
        set background permittivity

    Returns
    -------
    PyEITMesh
        mesh object
    """
    if isinstance(anomaly, PyEITAnomaly_Circle):
        anomaly = [anomaly]
    if isinstance(mesh.perm, np.ndarray):
        perm = mesh.perm.copy()
    else:
        perm = mesh.perm * np.ones(mesh.n_elems)
    # reset background if needed
    if background is not None:
        perm = background * np.ones(mesh.n_elems)

    # complex-valued permmitivity
    for an in anomaly:
        if np.iscomplex(an.perm):
            perm = perm.astype("complex")
            break

    # assign anomaly values (for elements in regions)
    tri_centers = mesh.elem_centers
    for an in anomaly:
        mask = an.mask(tri_centers)
        perm[mask] = an.perm

    return PyEITMesh(
        node=mesh.node,
        element=mesh.element,
        perm=perm,
        el_pos=mesh.el_pos,
        ref_node=mesh.ref_node,
    )


# ---------------------------------------------------------------------------
# Protocol data structures (adapted from pyeit/eit/protocol.py)
# ---------------------------------------------------------------------------


@dataclass
class PyEITProtocol:
    """
    EIT Protocol buid-in protocol object

    Parameters
    ----------
    ex_mat: np.ndarray
        excitation matrix (pairwise)
    meas_mat: np.ndarray
        measurement matrix (differential pairs), support inhomogeneous number
        of measurements per excitation pair.
    keep_ba: np.ndarray
        boolean array index for keeping measurements
    """

    ex_mat: np.ndarray
    meas_mat: np.ndarray
    keep_ba: np.ndarray

    def __post_init__(self) -> None:
        """Checking of the inputs"""
        self.ex_mat = self._check_ex_mat(self.ex_mat)
        self.meas_mat = self._check_meas_mat(self.meas_mat)
        self.keep_ba = self._check_keep_mat(self.keep_ba)

    def _check_ex_mat(self, ex_mat: np.ndarray) -> np.ndarray:
        """
        Check/init stimulation

        Parameters
        ----------
        ex_mat : np.ndarray
            stimulation/excitation matrix, of shape (n_exc, 2).
            If single stimulation (ex_line) is passed only a list of length 2
            and np.ndarray of size 2 will be treated.

        Returns
        -------
        np.ndarray
            stimulation matrix

        Raises
        ------
        TypeError
            Only accept, list of length 2, np.ndarray of size 2,
            or np.ndarray of shape (n_exc, 2)
        """
        if isinstance(ex_mat, list) and len(ex_mat) == 2:
            # case ex_line has been passed instead of ex_mat
            ex_mat = np.array([ex_mat]).reshape((1, 2))  # build a 2D array
        elif isinstance(ex_mat, np.ndarray) and ex_mat.size == 2:
            # case ex_line np.ndarray has been passed instead of ex_mat
            ex_mat = ex_mat.reshape((-1, 2))

        if not isinstance(ex_mat, np.ndarray):
            raise TypeError(f"Wrong type of {type(ex_mat)=}, expected an ndarray;")
        if ex_mat.ndim != 2 or ex_mat.shape[1] != 2:
            raise TypeError(f"Wrong shape of {ex_mat.shape=}, should be (n_exc, 2);")

        return ex_mat

    def _check_meas_mat(self, meas_mat: np.ndarray) -> np.ndarray:
        """
        Check measurement pattern

        Parameters
        ----------
        meas_mat : np.ndarray
           measurements pattern / subtract_row pairs [N, M]; shape (n_meas_tot, 3)

        Returns
        -------
        np.ndarray
            measurements pattern / subtract_row pairs [N, M]; shape (n_meas_tot, 3)

        Raises
        ------
        TypeError
            raised if meas_mat is not a np.ndarray of shape (n_meas_tot, 3)
        """
        if not isinstance(meas_mat, np.ndarray):
            raise TypeError(f"Wrong type of {type(meas_mat)=}, expected an ndarray;")
        if meas_mat.ndim != 2 or meas_mat.shape[-1] != 3:
            raise TypeError(f"{meas_mat.shape=} must be (n_meas_tot, 3);")

        return meas_mat

    def _check_keep_mat(self, keep_ba: np.ndarray) -> np.ndarray:
        """check keep boolean array"""
        if not isinstance(keep_ba, np.ndarray):
            raise TypeError(f"Wrong type of {type(keep_ba)=}, expected an ndarray;")

        return keep_ba

    @property
    def n_exc(self) -> int:
        """
        Returns
        -------
        int
            number of excitation
        """
        return self.ex_mat.shape[0]

    @property
    def n_meas(self) -> int:
        """
        Returns
        -------
        int
            total amount of measurements (n_meas_tot)
        """
        return self.meas_mat.shape[0]

    @property
    def n_el(self) -> int:
        """
        Returns
        -------
        int
            infer the number of electrodes used in the excitation and
            measurements patterns, where the electrodes are numbered [0, n_el-1].
        """
        return (
            int(max(max(self.ex_mat.flatten()), max(self.meas_mat[:, :-1].flatten())))
            + 1
        )


# ---------------------------------------------------------------------------
# Protocol creation functions (adapted from pyeit/eit/protocol.py)
# ---------------------------------------------------------------------------


def build_exc_pattern_std(n_el: int = 16, dist: int = 1) -> np.ndarray:
    """
    Generate scan matrix, `ex_mat` ( or excitation pattern), see notes

    Parameters
    ----------
    n_el : int, optional
        number of electrodes, by default 16
    dist : int, optional
        distance (number of electrodes) of A to B, by default 1
        For 'adjacent'- or 'neighbore'-mode (default) use `1` , and
        for 'apposition'-mode use `n_el/2` (see Examples).

    Returns
    -------
    np.ndarray
        stimulation matrix; shape (n_exc, 2)

    Notes
    -----
        - in the scan of EIT (or stimulation matrix), we use 4-electrodes
        mode, where A, B are used as positive and negative stimulation
        electrodes and M, N are used as voltage measurements.
        - `1` (A) for positive current injection, `-1` (B) for negative current
        sink

    Examples
    --------
        n_el=16
        if mode=='neighbore':
            ex_mat = build_exc_pattern_std(n_el=n_el)
        elif mode=='apposition':
            ex_mat = build_exc_pattern_std(dist=n_el/2)

    WARNING
    -------
        `ex_mat` is a local index, where it is ranged from 0...15, within the
        range of the number of electrodes. In FEM applications, you should
        convert `ex_mat` to global index using the (global) `el_pos` parameters.
    """
    return np.array([[i, np.mod(i + dist, n_el)] for i in range(n_el)])


def build_meas_pattern_std(
    ex_mat: np.ndarray,
    n_el: int = 16,
    step: int = 1,
    parser: Union[str, List[str]] = "std",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the measurement pattern (subtract_row-voltage pairs [N, M]) for
    all excitations on boundary electrodes. The excitation index (exc_id)
    are also recorded for computing subtract_row_vectorized and smear_nd.

    we direct operate on measurements or Jacobian on electrodes,
    so, we can use LOCAL index in this module, do not require el_pos.

    Notes
    -----
    ABMN Model.
    A: current driving electrode,
    B: current sink,
    M, N: boundary electrodes, where v_diff = v_n - v_m.

    Parameters
    ----------
    ex_mat : np.ndarray
        Nx2 array, [positive electrode, negative electrode]. ; shape (n_exc, 2)
    n_el : int, optional
        number of total electrodes, by default 16
    step : int, optional
        measurement method (two adjacent electrodes are used for measuring),
        by default 1 (adjacent)
    parser : Union[str, List[str]], optional
        parsing the format of each frame in measurement/file, by default 'std'
        if parser contains 'fmmu', or 'rotate_meas' then data are trimmed,
        boundary voltage measurements are re-indexed and rotated,
        start from the positive stimulus electrode start index 'A'.
        if parser contains 'std', or 'no_rotate_meas' then data are trimmed,
        the start index (i) of boundary voltage measurements is always 0.
        if parser contains 'meas_current', the measurements on current carrying
        electrodes are allowed. Otherwise the measurements on current carrying
        electrodes are discarded (like 'no_meas_current' option in EIDORS3D).

    Returns
    -------
    diff_op: np.ndarray
        measurements pattern / subtract_row pairs, and the excitation indice;
        shape (n_meas_tot, 3), for each row, it represents [Ni, Mi, exc_id]
    keep_ba: np.ndarray
        (n_meas_tot,) boolean array
    """
    if not isinstance(parser, list):  # transform parser into list
        parser = [parser]
    meas_current = "meas_current" in parser
    fmmu_rotate = any(p in ("fmmu", "rotate_meas") for p in parser)

    diff_op, keep_ba = [], []
    for exc_id, exc_line in enumerate(ex_mat):
        a, b = exc_line[0], exc_line[1]
        i0 = a if fmmu_rotate else 0
        # build [[m, n, idx]_i] array
        m = (i0 + np.arange(n_el)) % n_el
        n = (m + step) % n_el
        idx = exc_id * np.ones(n_el)
        meas_pattern = np.vstack([n, m, idx]).T

        diff_keep = np.logical_and.reduce((m != a, m != b, n != a, n != b))
        keep_ba.append(diff_keep)
        if not meas_current:
            meas_pattern = meas_pattern[diff_keep]
        diff_op.append(meas_pattern.astype(int))

    return np.vstack(diff_op), np.array(keep_ba).ravel()


def create_protocol(
    n_el: int = 16,
    dist_exc: Union[int, List[int]] = 1,
    step_meas: int = 1,
    parser_meas: Union[str, List[str]] = "std",
) -> PyEITProtocol:
    """
    Return an EIT protocol, comprising an excitation and a measurement pattern

    Parameters
    ----------
    n_el : int, optional
        number of total electrodes, by default 16
    dist_exc : Union[int, List[int]], optional
        distance (number of electrodes) of A to B, by default 1
        For 'adjacent'- or 'neighbore'-mode (default) use `1` , and
        for 'apposition'-mode use `n_el/2`. (see `build_exc_pattern`)
        if a list of integer is passed the excitation will be stacked together.
    step_meas : int, optional
        measurement method (two adjacent electrodes are used for measuring),
        by default 1 (adjacent). (see `build_meas_pattern`)
    parser_meas : Union[str, List[str]], optional
        parsing the format of each frame in measurement/file, by default 'std'.
        (see `build_meas_pattern`)

    Returns
    -------
    PyEITProtocol
        EIT protocol object

    Raises
    ------
    TypeError
        if dist_exc is not list or an int
    """
    if isinstance(dist_exc, int):
        dist_exc = [dist_exc]

    if not isinstance(dist_exc, list):
        raise TypeError(f"{type(dist_exc)=} should be a List[int]")

    _ex_mat = [build_exc_pattern_std(n_el, dist) for dist in dist_exc]
    ex_mat = np.vstack(_ex_mat)

    meas_mat, keep_ba = build_meas_pattern_std(ex_mat, n_el, step_meas, parser_meas)
    return PyEITProtocol(ex_mat, meas_mat, keep_ba)


# ---------------------------------------------------------------------------
# FEM functions (adapted from pyeit/eit/fem.py)
# ---------------------------------------------------------------------------


def det2x2(s1: np.ndarray, s2: np.ndarray):
    """Calculate the determinant of a 2x2 matrix"""
    return s1[0] * s2[1] - s1[1] * s2[0]


def _k_triangle(xy: np.ndarray):
    """
    Given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy: np.ndarray
        (x,y) of nodes 1,2,3 given in counterclockwise manner

    Returns
    -------
    np.ndarray
        local stiffness matrix
    """
    # edges (vector) of triangles
    s = xy[[2, 0, 1]] - xy[[1, 2, 0]]

    # area of triangles. Note, abs is removed since version 2020,
    # user must make sure all triangles are CCW (conter clock wised).
    # at = 0.5 * np.linalg.det(s[[0, 1]])
    at = 0.5 * det2x2(s[0], s[1])

    # Local stiffness matrix (e for element)
    return np.dot(s, s.T) / (4.0 * at)


def _k_tetrahedron(xy: np.ndarray):
    """
    Given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy: np.ndarray
        (x,y) of nodes 1, 2, 3, 4 given in counterclockwise manner, see notes.

    Returns
    -------
    np.ndarray
        local stiffness matrix

    Notes
    -----
    A tetrahedron is described using [0, 1, 2, 3] (local node index) or
    [171, 27, 9, 53] (global index). Counterclockwise (CCW) is defined
    such that the barycentric coordinate of face (1->2->3) is positive.
    """
    s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]

    # volume of the tetrahedron, Note abs is removed since version 2020,
    # user must make sure all tetrahedrons are CCW (counter clock wised).
    vt = 1.0 / 6 * la.det(s[[0, 1, 2]])

    # calculate area (vector) of triangle faces
    # re-normalize using alternative (+,-) signs
    ij_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
    signs = [1, -1, 1, -1]
    a = np.array([sign * np.cross(s[i], s[j]) for (i, j), sign in zip(ij_pairs, signs)])

    # local (e for element) stiffness matrix
    return np.dot(a, a.transpose()) / (36.0 * vt)


def calculate_ke(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """
    Calculate local stiffness matrix on all elements.

    Parameters
    ----------
    pts: np.ndarray
        Nx2 (x,y) or Nx3 (x,y,z) coordinates of points
    tri: np.ndarray
        Mx3 (triangle) or Mx4 (tetrahedron) connectivity of elements

    Returns
    -------
    np.ndarray
        n_tri x (n_dim x n_dim) 3d matrix
    """
    n_tri, n_vertices = tri.shape

    # check dimension
    # '3' : triangles
    # '4' : tetrahedrons
    if n_vertices == 3:
        _k_local = _k_triangle
    elif n_vertices == 4:
        _k_local = _k_tetrahedron
    else:
        raise TypeError("The num of vertices of elements must be 3 or 4")

    # default data types for ke
    ke_array = np.zeros((n_tri, n_vertices, n_vertices))
    for ei in range(n_tri):
        no = tri[ei, :]
        xy = pts[no]

        # compute the KIJ (permittivity=1.)
        ke = _k_local(xy)
        ke_array[ei] = ke

    return ke_array


def assemble(
    ke: np.ndarray, tri: np.ndarray, perm: np.ndarray, n_pts: int, ref: int = 0
):
    """
    Assemble the stiffness matrix (using sparse matrix)

    Parameters
    ----------
    ke: np.ndarray
        n_tri x (n_dim x n_dim) 3d matrix
    tri: np.ndarray
        the structure of mesh
    perm: np.ndarray
        n_tri x 1 conductivities on elements
    n_pts: int
        number of nodes
    ref: int, optional
        reference electrode, by default 0

    Returns
    -------
    np.ndarray
        NxN array of complex stiffness matrix

    Notes
    -----
    you may use sparse matrix (IJV) format to automatically add the local
    stiffness matrix to the global matrix.
    """
    n_tri, n_vertices = tri.shape

    # New: use IJV indexed sparse matrix to assemble K (fast, prefer)
    # index = np.array([np.meshgrid(no, no, indexing='ij') for no in tri])
    # note: meshgrid is slow, using handcraft sparse index, for example
    # let tri=[[1, 2, 3], [4, 5, 6]], then indexing='ij' is equivalent to
    # row = [1, 1, 1, 2, 2, 2, ...]
    # col = [1, 2, 3, 1, 2, 3, ...]
    row = np.repeat(tri, n_vertices).ravel()
    col = np.repeat(tri, n_vertices, axis=0).ravel()
    data = np.array([ke[i] * perm[i] for i in range(n_tri)]).ravel()

    # set reference nodes before constructing sparse matrix
    if 0 <= ref < n_pts:
        dirichlet_ind = np.logical_or(row == ref, col == ref)
        # K[ref, :] = 0, K[:, ref] = 0
        row = row[~dirichlet_ind]
        col = col[~dirichlet_ind]
        data = data[~dirichlet_ind]
        # K[ref, ref] = 1.0
        row = np.append(row, ref)
        col = np.append(col, ref)
        data = np.append(data, 1.0)

    # for efficient sparse inverse (csc)
    return sparse.csr_matrix((data, (row, col)), shape=(n_pts, n_pts))


def subtract_row_vectorized(v: np.ndarray, meas_pattern: np.ndarray):
    """
    Build the voltage differences on axis=1 using the meas_pattern.
    v_diff[k] = v[exc_id, i] - v[exc_id, j]

    New implementation 33% less computation time

    Parameters
    ----------
    v: np.ndarray
        (n_exc, n_el) boundary measurements or (n_exc, (n_el, n_element)) nodes resistance
    meas_pattern: np.ndarray
        Nx2 subtract_row pairs; shape (n_meas_tot, 3)

    Returns
    -------
    np.ndarray
        difference measurements v_diff
    """
    idx = meas_pattern[:, 2]
    return v[idx, meas_pattern[:, 0]] - v[idx, meas_pattern[:, 1]]


def _smear_nd(f: np.ndarray, fb: np.ndarray, meas_pattern: np.ndarray) -> np.ndarray:
    """
    Build smear matrix B for bp (vectorized version using exc_idx from meas_pattern)

    Parameters
    ----------
    f: np.ndarray
        potential on nodes; shape (n_exc, n_pts)
    fb: np.ndarray
        potential on adjacent electrodes; shape (n_exc, n_el)
    meas_pattern: np.ndarray
        electrodes numbering pairs; shape (n_meas_tot, 3)

    Returns
    -------
    np.ndarray
        back-projection (smear) matrix; shape (n_meas_tot, n_pts), dtype= bool
    """
    n = meas_pattern[:, 0]
    m = meas_pattern[:, 1]
    exc_id = meas_pattern[:, 2]
    # (n_meas_tot,) voltages on electrodes
    f_min = np.minimum(fb[exc_id, n], fb[exc_id, m])
    f_max = np.maximum(fb[exc_id, n], fb[exc_id, m])
    # contruct matrix of shapes (n_meas_tot, n_pts) for comparison
    n_pts = f.shape[1]
    f_min = np.repeat(f_min[:, np.newaxis], n_pts, axis=1)
    f_max = np.repeat(f_max[:, np.newaxis], n_pts, axis=1)
    f_pts = f[exc_id]  # voltages on nodes of all excitations

    return np.array((f_min < f_pts) & (f_pts <= f_max))


# ---------------------------------------------------------------------------
# Forward solver classes (adapted from pyeit/eit/fem.py)
# ---------------------------------------------------------------------------


class Forward:
    """FEM forward computing code"""

    def __init__(self, mesh: PyEITMesh) -> None:
        """
        FEM forward solver.
        A good FEM forward solver should only depend on
        mesh structure and the position of electrodes.

        Parameters
        ----------
        mesh: PyEITMesh
            mesh object

        Note
        ----
        The nodes are continuous numbered, the numbering of an element is
        CCW (counter-clock-wise).
        """
        self.mesh = mesh
        # coefficient matrix [initialize]
        self.se = calculate_ke(self.mesh.node, self.mesh.element)
        self.assemble_pde(self.mesh.perm)

    def assemble_pde(
        self, perm: Optional[Union[int, float, complex, np.ndarray]] = None
    ) -> None:
        """
        assemble PDE

        Parameters
        ----------
        perm : Union[int, float, np.ndarray]
            permittivity on elements ; shape (n_tri,).
            if `None`, assemble_pde is aborded

        """
        if perm is None:
            return
        perm_array = self.mesh.get_valid_perm_array(perm)
        self.kg = assemble(
            self.se,
            self.mesh.element,
            perm_array,
            self.mesh.n_nodes,
            ref=self.mesh.ref_node,
        )

    def solve(self, ex_line: np.ndarray = np.array([0, 1])):
        """
        Calculate and compute the potential distribution (complex-valued)
        corresponding to the permittivity distribution `perm ` for a
        excitation contained specified by `ex_line` (Neumann BC)

        Parameters
        ----------
        ex_line : np.ndarray, optional
            stimulation/excitation matrix, of shape (2,)

        Returns
        -------
        np.ndarray
            potential on nodes ; shape (n_pts,)

        Notes
        -----
        Currently, only simple electrode model is supported,
        CEM (complete electrode model) is under development.
        """
        # using natural boundary conditions
        b = np.zeros(self.mesh.n_nodes)
        b[self.mesh.el_pos[ex_line]] = [1, -1]

        # solve
        return scipy.sparse.linalg.spsolve(self.kg, b)

    def solve_vectorized(self, ex_mat: np.ndarray) -> np.ndarray:
        """
        Calculate and compute the potential distribution (complex-valued)
        corresponding to the permittivity distribution `perm ` for a
        excitation contained specified by `ex_mat` (Neumann BC)

        Parameters
        ----------
        ex_mat : np.ndarray
            stimulation/excitation matrix, of shape (n_exc, 2)

        Returns
        -------
        np.ndarray
            potential on nodes ; shape (n_exc, n_pts)

        Notes
        -----
        Currently, only simple electrode model is supported,
        CEM (complete electrode model) is under development.
        """

        # using natural boundary conditions
        b = np.zeros((ex_mat.shape[0], self.mesh.n_nodes))
        b[np.arange(b.shape[0])[:, None], self.mesh.el_pos[ex_mat]] = [1, -1]
        result = np.empty((ex_mat.shape[0], self.kg.shape[0]), dtype=complex)

        # TODO Need to inspect this deeper
        for i in range(result.shape[0]):
            result[i] = sparse.linalg.spsolve(self.kg, b[i])

        # solve
        return result


class EITForward(Forward):
    """EIT Forward simulation, depends on mesh and protocol"""

    def __init__(self, mesh: PyEITMesh, protocol: PyEITProtocol) -> None:
        """
        EIT Forward Solver

        Parameters
        ----------
        mesh: PyEITMesh
            mesh object
        protocol: PyEITProtocol
            measurement object

        Notes
        -----
        The Jacobian and the boundary voltages used the SIGN information,
        for example, V56 = V6 - V5 = -V65. If you are using absolute boundary
        voltages for imaging, you MUST normalize it with the signs of v0
        under each current-injecting pattern.
        """
        self._check_mesh_protocol_compatibility(mesh, protocol)

        # FEM solver
        super().__init__(mesh=mesh)

        # EIT measurement protocol
        self.protocol = protocol

    def _check_mesh_protocol_compatibility(
        self, mesh: PyEITMesh, protocol: PyEITProtocol
    ) -> None:
        """
        Check if mesh and protocol are compatible

        - #1 n_el in mesh >=  n_el in protocol
        - #2 .., TODO if necessary

        Raises
        ------
        ValueError
            if protocol is not compatible to the mesh
        """
        # n_el in mesh should be >=  n_el in protocol
        m_n_el = mesh.n_el
        p_n_el = protocol.n_el

        if m_n_el != p_n_el:
            warnings.warn(
                f"The mesh use {m_n_el} electrodes, and the protocol use only {p_n_el} electrodes",
                stacklevel=2,
            )

    def solve_eit(
        self,
        perm: Optional[Union[int, float, complex, np.ndarray]] = None,
    ):
        """
        EIT simulation, generate forward v measurements

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of forward v measurements will be
            based on the permittivity of the mesh, self.mesh.perm
        Returns
        -------
        v: np.ndarray
            simulated boundary voltage measurements; shape(n_exe*n_el,)
        """
        self.assemble_pde(perm)
        f = self.solve_vectorized(self.protocol.ex_mat)
        v = subtract_row_vectorized(f[:, self.mesh.el_pos], self.protocol.meas_mat)

        return v.reshape(-1)

    def compute_jac(
        self,
        perm: Optional[Union[int, float, complex, np.ndarray]] = None,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Jacobian matrix and initial boundary voltage meas.
        extimation v0

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of Jacobian matrix will be based
            on the permittivity of the mesh, self.mesh.perm
        normalize : bool, optional
            flag for Jacobian normalization, by default False.
            If True the Jacobian is normalized

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Jacobian matrix, initial boundary voltage meas. extimation v0

        """
        # update k if necessary and calculate r=inv(k), dense matrix, slow
        self.assemble_pde(perm)
        r_mat = la.inv(self.kg.toarray())[self.mesh.el_pos]
        r_el = np.full((self.protocol.ex_mat.shape[0],) + r_mat.shape, r_mat)
        # nodes potential
        f = self.solve_vectorized(self.protocol.ex_mat)
        f_el = f[:, self.mesh.el_pos]
        # build measurements and node resistance
        v = subtract_row_vectorized(f_el, self.protocol.meas_mat)
        ri = subtract_row_vectorized(r_el, self.protocol.meas_mat)
        v0 = v.reshape(-1)

        # Build Jacobian matrix element wise (column wise)
        # Je = Re*Ke*Ve = (n_measx3) * (3x3) * (3x1)
        jac = np.zeros((self.protocol.n_meas, self.mesh.n_elems), dtype=self.mesh.dtype)
        indices = self.protocol.meas_mat[:, 2]
        f_n = f[indices]  # replica of potential on nodes of difference excitations
        for e, ijk in enumerate(self.mesh.element):
            jac[:, e] = np.sum(np.dot(ri[:, ijk], self.se[e]) * f_n[:, ijk], axis=1)

        # Jacobian normalization: divide each row of J (J[i]) by abs(v0[i])
        if normalize:
            jac = jac / np.abs(v0[:, None])
        return jac, v0

    def compute_b_matrix(
        self,
        perm: Optional[Union[int, float, complex, np.ndarray]] = None,
    ):
        """
        Compute back-projection mappings (smear matrix)

        Parameters
        ----------
        perm : Union[int, float, np.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of smear matrix will be based
            on the permittivity of the mesh, self.mesh.perm

        Returns
        -------
        np.ndarray
            back-projection mappings (smear matrix); shape(n_exc, n_pts, 1), dtype= bool
        """
        self.assemble_pde(perm)
        f = self.solve_vectorized(self.protocol.ex_mat)
        f_el = f[:, self.mesh.el_pos]
        return _smear_nd(f, f_el, self.protocol.meas_mat)


# ---------------------------------------------------------------------------
# Interpolation utilities (adapted from pyeit/eit/interp2d.py)
# ---------------------------------------------------------------------------


def tri_area(pts: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """
    calculate the area of each triangle

    Parameters
    ----------
    pts: np.ndarray
        Nx2 array, (x,y) locations for points (can be also (x,y,z))
    sim: np.ndarray
        Mx3 array, elements (triangles) connectivity

    Returns
    -------
    a: np.ndarray
        Areas of triangles
    """
    pts_2D = pts[:, :2]  # get only x and y
    a = np.zeros(np.shape(sim)[0])
    for i, e in enumerate(sim):
        xy = pts_2D[e]
        # which can be simplified to
        # s = xy[[2, 0, 1]] - xy[[1, 2, 0]]
        s = xy[[2, 0]] - xy[[1, 2]]

        # a should be positive if triangles are CCW arranged
        a[i] = scipy.linalg.det(s)

    return a * 0.5


def tet_volume(pts: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """
    calculate the volume of each tetrahedron

    Parameters
    ----------
    pts: np.ndarray
        Nx3 array, (x,y, z) locations for points
    sim: np.ndarray
        Mx4 array, elements (tetrahedrons) connectivity

    Returns
    -------
    v: np.ndarray
        Volumes of tetrahedrons
    """
    v = np.zeros(np.shape(sim)[0])
    for i, e in enumerate(sim):
        xyz = pts[e]
        s = xyz[[2, 3, 0]] - xyz[[1, 2, 3]]

        # a should be positive if triangles are CCW arranged
        v[i] = scipy.linalg.det(s)

    return v / 6.0


def sim2pts(pts: np.ndarray, sim: np.ndarray, sim_values: np.ndarray):
    """
    (2D/3D) compatible.

    Interp values on points using values on simplex,
    a simplex can be triangle or tetrahedron.
    The areas/volumes are used as weights.

    f_n = (sum_e r_e*S_e) / (sum_e S_e)

    where r_e is the value on triangles who share the node n,
    S_e is the area of triangle e.

    Parameters
    ----------
    pts: np.ndarray
        Nx2 or Nx3 array, coordinates of points
    sim: np.ndarray
        Mx3, Mx4 array, elements or simplex
        triangles denote connectivity [[i, j, k]]
        tetrahedrons denote connectivity [[i, j, m, n]]
    sim_values: np.ndarray
        Mx1 array, values on elements

    Notes
    -----
    This function is similar to pdeprtni of MATLAB pde.
    """
    N = pts.shape[0]
    M, dim = sim.shape
    # calculate the weights
    # triangle/tetrahedron must be CCW (recommended), then a is positive
    if dim == 3:
        weight_func = tri_area
    elif dim == 4:
        weight_func = tet_volume
    else:
        raise TypeError("The num of vertices of elements must be 3 or 4")
    weights = weight_func(pts, sim)
    # build tri->pts matrix, could be accelerated using sparse matrix
    row = np.ravel(sim)
    col = np.repeat(np.arange(M), dim)  # [0, 0, 0, 1, 1, 1, ...]
    data = np.repeat(weights, dim)
    e2n_map = coo_matrix((data, (row, col)), shape=(N, M)).tocsr()
    # map values from elements to nodes
    # and re-weight by the sum of the areas/volumes of adjacent elements
    f = e2n_map.dot(sim_values)
    w = np.sum(e2n_map.toarray(), axis=1)

    return f / w


# ---------------------------------------------------------------------------
# Convenience wrapper class
# ---------------------------------------------------------------------------


class EITForwardModel:
    """Convenience wrapper combining mesh, protocol, and FEM solver."""

    def __init__(self, mesh: PyEITMesh, protocol: PyEITProtocol):
        self.mesh = mesh
        self.protocol = protocol
        self.fwd = EITForward(mesh, protocol)

    def solve_eit(self, perm=None):
        """
        EIT simulation, generate forward v measurements.

        Parameters
        ----------
        perm : optional
            permittivity on elements; if None uses mesh default.

        Returns
        -------
        np.ndarray
            simulated boundary voltage measurements
        """
        return self.fwd.solve_eit(perm)

    def compute_jac(self, perm=None, normalize=False):
        """
        Compute the Jacobian matrix and initial boundary voltage.

        Parameters
        ----------
        perm : optional
            permittivity on elements; if None uses mesh default.
        normalize : bool
            flag for Jacobian normalization.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Jacobian matrix and initial voltage v0
        """
        return self.fwd.compute_jac(perm, normalize)

    def compute_b_matrix(self, perm=None):
        """
        Compute back-projection mappings (smear matrix).

        Parameters
        ----------
        perm : optional
            permittivity on elements; if None uses mesh default.

        Returns
        -------
        np.ndarray
            back-projection (smear) matrix
        """
        return self.fwd.compute_b_matrix(perm)
