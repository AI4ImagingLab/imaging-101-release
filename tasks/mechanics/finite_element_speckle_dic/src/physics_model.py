"""Q4 integrated image kinematics, adapted from muDIC (MIT, PolymerGuy 2019)."""
import numpy as np
from scipy.ndimage import spline_filter, map_coordinates


def shape_functions(xi, eta):
    """Bilinear shape values (...,4), parent coordinates in [0,1]^2."""
    return np.stack(((1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta), axis=-1)


def element_pixels(corners):
    """Integer reference pixels, including both element edges, and Q4 values."""
    lo = np.floor(corners.min(axis=0)).astype(int)
    hi = np.ceil(corners.max(axis=0)).astype(int)
    x, y = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
    x, y = x.ravel(), y.ravel()
    # The bound acquisition has an axis-aligned regular reference mesh.
    xi = (x-corners[0, 0])/(corners[1, 0]-corners[0, 0])
    eta = (y-corners[0, 1])/(corners[3, 1]-corners[0, 1])
    return x, y, shape_functions(xi, eta)


def prepare_reference(image, nodes, elements):
    """Assemble fixed reference-image Jacobian blocks and inverse normal matrix."""
    blocks, hessian = [], np.zeros((2*len(nodes), 2*len(nodes)))
    for element in elements:
        x, y, N = element_pixels(nodes[element])
        gx = (image[y, x+1]-image[y, x-1])/2
        gy = (image[y+1, x]-image[y-1, x])/2
        B = np.vstack(((N*gx[:, None]).T, (N*gy[:, None]).T))
        dofs = np.r_[2*element, 2*element+1]
        hessian[np.ix_(dofs, dofs)] += B @ B.T
        blocks.append((N, B, image[y, x].copy()))
    return blocks, np.linalg.inv(hessian)


def filtered_frames(image, nodes, elements, pad=10, order=4):
    """Preserve upstream full-image spline prefilter then per-element crop."""
    filtered = spline_filter(image, order=order)
    frames = []
    for element in elements:
        lo = np.floor(nodes[element].min(axis=0)).astype(int)
        hi = np.ceil(nodes[element].max(axis=0)).astype(int)
        frames.append((filtered[lo[1]-pad:hi[1]+pad, lo[0]-pad:hi[0]+pad], lo))
    return frames


def sample_element(frame, lower, N, current_nodes, pad=10, order=4):
    """Evaluate deformed intensity at current Q4 mapped reference pixels."""
    xy = N @ (current_nodes-lower+pad)
    return map_coordinates(frame, xy.T[::-1], order=order, prefilter=False)


def green_strain(nodes, displaced_nodes, elements):
    """Element-center Green-Lagrange strain (E,2,2), dimensionless."""
    gradients = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    values = []
    for element in elements:
        reference_jacobian = nodes[element].T @ gradients
        current_jacobian = displaced_nodes[element].T @ gradients
        F = current_jacobian @ np.linalg.inv(reference_jacobian)
        values.append((F.T @ F-np.eye(2))/2)
    return np.array(values)
