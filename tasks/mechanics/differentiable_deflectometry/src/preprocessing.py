"""
Preprocessing for differentiable refractive deflectometry.

Handles calibration loading, fringe analysis, and extraction of measured
intersection points from phase-shifting images.

Extracted from: diffmetrology/solvers.py (Fringe class), diffmetrology/utils.py
(DiffMetrology calibration/preprocessing methods), and demo_experiments.py.
"""
import numpy as np
import scipy.io
import scipy.ndimage
import torch
from matplotlib.image import imread
from skimage.restoration import unwrap_phase

from .physics_model import (
    Camera, Screen, Lensgroup, Scene, Transformation, PrettyPrinter,
    set_texture,
)


# ===========================================================================
# Fringe analysis (from solvers.py lines 7-73)
# ===========================================================================

class Fringe(PrettyPrinter):
    """Four-step phase-shifting fringe analysis to resolve displacements."""

    def __init__(self):
        self.PHASE_SHIFT_COUNT = 4
        self.XY_COUNT = 2

    def solve(self, fs):
        """Solve for amplitude, modulation, and phase from phase-shifted images.

        Returns: (a, b, psi) where each has shape [2, ...original_dims...]
        """
        def single(fs):
            ax, bx, psix = self._solve(fs[0:self.PHASE_SHIFT_COUNT])
            ay, by, psiy = self._solve(fs[self.PHASE_SHIFT_COUNT:self.XY_COUNT * self.PHASE_SHIFT_COUNT])
            return np.array([ax, ay]), np.array([bx, by]), np.array([psix, psiy])

        fsize = list(fs.shape)
        xy_index = fsize.index(self.XY_COUNT * self.PHASE_SHIFT_COUNT)
        inds = [i for i in range(len(fsize))]
        inds.remove(xy_index)
        inds = [xy_index] + inds

        a, b, p = single(fs.transpose(inds))
        return a, b, p

    def unwrap(self, fs, Ts, valid=None):
        """Unwrap phase and convert to displacement in mm."""
        print('unwraping ...')
        if valid is None:
            valid = 1.0
        F = valid * fs
        fs_unwrapped = np.zeros(fs.shape)
        for xy in range(fs.shape[0]):
            print(f'xy = {xy} ...')
            for T in range(fs.shape[1]):
                print(f't = {T} ...')
                t = Ts[T]
                for i in range(fs.shape[2]):
                    fs_unwrapped[xy, T, i] = unwrap_phase(F[xy, T, i, ...]) * t / (2 * np.pi)
        return fs_unwrapped

    @staticmethod
    def _solve(fs):
        """Four-step phase shifting: f(x) = a + b cos(phi + psi)."""
        a = np.mean(fs, axis=0)
        b = 0.0
        for f in fs:
            b += (f - a)**2
        b = b / 2.0
        psi = np.arctan2(fs[3] - fs[1], fs[0] - fs[2])
        return a, b, psi


# ===========================================================================
# Calibration loading (from utils.py)
# ===========================================================================

def load_calibration(calibration_path, rotation_path, lut_path=None, scale=1, device=torch.device('cpu')):
    """Load camera/screen calibration from MATLAB .mat files.

    Args:
        calibration_path: Directory containing cams.mat, cam1.mat, cam2.mat, checkerboard.png
        rotation_path: Path to rotation.mat
        lut_path: Optional path to gammas.mat
        scale: Scale factor for calibration parameters
        device: torch device

    Returns:
        cameras: list of Camera objects
        screen: Screen object
        p_rotation: rotation calibration points (Tensor)
        lut_data: optional LUT data dict (or None)
    """
    mat_g = scipy.io.loadmat(calibration_path + '/cams.mat')

    # Parse cameras
    filmsize = (mat_g['filmsize'] * scale).astype(int)
    f = mat_g['f'] * scale
    c = mat_g['c'] * scale
    k = mat_g['k'] * scale
    p = mat_g['p'] * scale
    R = mat_g['R']
    t = mat_g['t']

    def matlab2ours(R, t):
        return R, -R @ t

    cameras = [Camera(
        Transformation(*matlab2ours(R[..., i], t[..., i])),
        filmsize[i], f[i], c[i], k[i], p[i], device
    ) for i in range(len(f))]

    # Parse screen
    pixelsize = 1e-3 * mat_g['display_pixel_size'][0][0]
    size = pixelsize * np.array([1600, 2560])
    im = imread(calibration_path + '/checkerboard.png')
    im = np.mean(im, axis=-1)

    screen = Screen(Transformation(np.eye(3), np.zeros(3)), size, pixelsize, im, device)

    # Rotation calibration
    mat_r = scipy.io.loadmat(rotation_path)
    p_rotation = torch.Tensor(np.stack((mat_r['p1'][0], mat_r['p2'][0]), axis=-1)).T

    # LUT data
    lut_data = None
    if lut_path is not None:
        mat_lut = scipy.io.loadmat(lut_path)
        lut_data = {
            'lut': mat_lut['Js'][:, :len(cameras)],
            'bbd': mat_lut['bs'].reshape((2, len(cameras)))
        }

    return cameras, screen, p_rotation, lut_data


def compute_mount_origin(cameras, p_rotation, scale=1, device=torch.device('cpu'), verbose=True):
    """Estimate lens mount position by intersecting camera rays.

    Solves: min_{t1,t2} ||L1(t1) - L2(t2)||^2 via least squares.

    Args:
        cameras: list of Camera objects
        p_rotation: rotation calibration pixel coordinates (Tensor)
        device: torch device

    Returns:
        origin: mount origin as numpy array [3]
    """
    N = len(cameras)
    rays = [cameras[i].sample_ray(
        (p_rotation[i] * scale)[None, None, ...].to(device), is_sampler=False) for i in range(N)]

    t, r = np.linalg.lstsq(
        torch.stack((rays[0].d, -rays[1].d), axis=-1).cpu().detach().numpy(),
        -(rays[0].o - rays[1].o).cpu().detach().numpy(), rcond=None
    )[0:2]

    t_pt = torch.Tensor(t).to(device)
    os = [rays[i](t_pt[i]) for i in range(N)]
    if verbose:
        for i, o in enumerate(os):
            print('intersection point {}: {}'.format(i, o))
        print('|intersection points distance error| = {} mm'.format(np.sqrt(r[0])))
    return torch.mean(torch.stack(os), axis=0).cpu().detach().numpy()


# ===========================================================================
# Image preprocessing
# ===========================================================================

def crop_images(imgs, filmsize, full_filmsize=2048):
    """Center-crop raw images to the specified filmsize.

    Args:
        imgs: array with spatial dims matching full_filmsize
        filmsize: target [H, W] array
        full_filmsize: original image size (default 2048)

    Returns:
        cropped images
    """
    filmsize = np.array(filmsize)
    crop_offset = ((full_filmsize - filmsize) / 2).astype(int)
    return imgs[..., crop_offset[0]:crop_offset[0] + filmsize[0],
                     crop_offset[1]:crop_offset[1] + filmsize[1]]


def get_crop_offset(filmsize, full_filmsize=2048):
    """Compute the crop offset for a given filmsize."""
    filmsize = np.array(filmsize)
    return ((full_filmsize - filmsize) / 2).astype(int)


# ===========================================================================
# Intersection point solving (from utils.py DiffMetrology.solve_for_intersections)
# ===========================================================================

def solve_for_intersections(imgs, refs, Ts, scene, device=torch.device('cpu')):
    """Obtain measured intersection points from phase-shifting images.

    Pipeline: fringe solve -> unwrap -> valid map -> pixel-to-mm -> reference trace -> displacement

    Args:
        imgs: raw measurement images array
        refs: raw reference images array
        Ts: array of sinusoid periods
        scene: Scene object (with cameras and screen, no lensgroup needed)
        device: torch device

    Returns:
        ps_cap: measured intersection points (Tensor)
        valid_cap: validity mask (Tensor, bool)
        centers: center of valid region per camera (Tensor)
    """
    FR = Fringe()
    a_ref, b_ref, psi_ref = FR.solve(refs)
    a_cap, b_cap, psi_cap = FR.solve(imgs)

    camera_count = scene.camera_count

    def find_center(valid_cap):
        x, y = np.argwhere(valid_cap == 1).sum(0) / valid_cap.sum()
        return np.array([x, y])

    # Get valid map for each camera
    valid_map = []
    A = np.mean(a_cap, axis=(0, 1))
    B = np.mean(b_cap, axis=(0, 1))
    I = np.abs(np.mean(refs - imgs, axis=(0, 1)))
    for j in range(camera_count):
        thres = 0.07
        valid_ab = (A[j] > thres) & (B[j] > thres) & (I[j] < 2.0 * thres)
        label, num_features = scipy.ndimage.label(valid_ab)
        if num_features < 2:
            label_target = 1
        else:
            counts = np.array([np.count_nonzero(label == i) for i in range(num_features)])
            label_targets = np.where((200**2 < counts) & (counts < 500**2))[0]
            if len(label_targets) > 1:
                Dm = np.inf
                for l in label_targets:
                    c = find_center(label == l)
                    D = np.abs(scene.cameras[0].filmsize / 2 - c).sum()
                    if D < Dm:
                        Dm = D
                        label_target = l
            else:
                label_target = label_targets[0]
        V = label == label_target
        valid_map.append(V)
    valid_map = np.array(valid_map)

    psi_unwrap = FR.unwrap(psi_cap - psi_ref, Ts, valid=valid_map[None, None, ...])

    # Remove DC term
    k_DC = np.arange(-10, 11, 1)
    for t in range(len(Ts)):
        DCs = k_DC * Ts[t]
        psi_current = psi_unwrap[:, t, :, ...]
        psi_with_dc = valid_map[:, None, :, :, None] * (psi_current[..., None] + DCs[None, None, None, None, ...])
        DC_target = DCs[np.argmin(np.sum(psi_with_dc**2, axis=(2, 3)), axis=-1)]
        print("t = {}, DC_target =\n{}".format(Ts[t], DC_target))
        psi_unwrap[:, t, :, ...] += valid_map[None, :, ...] * np.transpose(DC_target, (0, 1))[..., None, None]

    # Convert from pixel to mm
    psi_x = psi_unwrap[0, ...] * scene.screen.pixelsize.item()
    psi_y = psi_unwrap[1, ...] * scene.screen.pixelsize.item()

    # Get median across Ts
    psi_x = np.mean(psi_x, axis=0)
    psi_y = np.mean(psi_y, axis=0)

    # Reference trace (no element)
    lensgroup_backup = scene.lensgroup
    scene.lensgroup = None
    ps_ref = torch.stack(scene.trace(with_element=False)[0])
    scene.lensgroup = lensgroup_backup

    ps_ref = valid_map[..., None] * ps_ref[..., 0:2].cpu().detach().numpy()

    # Final displacement
    p = ps_ref - np.stack((psi_y, psi_x), axis=-1)

    # Find valid map centers
    xs = []
    ys = []
    for i in range(len(valid_map)):
        x, y = np.argwhere(valid_map[i] == 1).sum(0) / valid_map[i].sum()
        xs.append(x)
        ys.append(y)
    centers = np.stack((np.array(xs), np.array(ys)), axis=-1)
    centers = np.fliplr(centers).copy()

    return (
        torch.Tensor(p).to(device),
        torch.Tensor(valid_map).bool().to(device),
        torch.Tensor(centers).to(device)
    )


def prepare_measurement_images(imgs, xs, valid_cap, fringe_a_cap, device=torch.device('cpu')):
    """Prepare background-subtracted measurement images for visualization.

    Args:
        imgs: raw measurement images
        xs: list of sinusoid indices to use
        valid_cap: validity mask (Tensor, bool)
        fringe_a_cap: DC component from fringe analysis
        device: torch device

    Returns:
        I0: normalized measurement images (Tensor)
    """
    imgs_sub = np.array([imgs[0, x, ...] for x in xs])
    imgs_sub = imgs_sub - fringe_a_cap[:, 0, ...]
    imgs_sub = np.sum(imgs_sub, axis=0)
    imgs_sub = valid_cap * torch.Tensor(imgs_sub).to(device)
    I0 = valid_cap * len(xs) * (imgs_sub - imgs_sub.min().item()) / (imgs_sub.max().item() - imgs_sub.min().item())
    return I0
