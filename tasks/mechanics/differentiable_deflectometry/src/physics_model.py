"""
Differentiable ray tracer for refractive deflectometry.

Merges the original diffmetrology package (basics.py, shapes.py, optics.py, scene.py)
into a single module preserving all original algorithm logic.

Reference: Wang et al., "Towards self-calibrated lens metrology by differentiable
refractive deflectometry", OSA Optics Express, 2021.
"""
import math
import re
from enum import Enum

import numpy as np
import torch
from scipy.interpolate import LSQBivariateSpline
from matplotlib.image import imread


# ===========================================================================
# Foundational types (from basics.py)
# ===========================================================================

class PrettyPrinter:
    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in vars(self).items():
            if val.__class__.__name__ in ('list', 'tuple'):
                for i, v in enumerate(val):
                    lines += '{}[{}]: {}'.format(key, i, v).split('\n')
            elif val.__class__.__name__ in 'dict':
                pass
            elif key == key.upper() and len(key) > 5:
                pass
            else:
                lines += '{}: {}'.format(key, val).split('\n')
        return '\n    '.join(lines)

    def to(self, device=torch.device('cpu')):
        for key, val in vars(self).items():
            if torch.is_tensor(val):
                setattr(self, key, val.to(device))
            elif isinstance(val, PrettyPrinter):
                val.to(device)
            elif isinstance(val, (list, tuple)):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        val[i] = v.to(device)
                    elif isinstance(v, PrettyPrinter):
                        v.to(device)


class Ray(PrettyPrinter):
    def __init__(self, o, d, wavelength, device=torch.device('cpu')):
        self.o = o
        self.d = d
        self.wavelength = wavelength
        self.mint = 1e-5
        self.maxt = 1e5
        self.to(device)

    def __call__(self, t):
        return self.o + t[..., None] * self.d


class Transformation(PrettyPrinter):
    def __init__(self, R, t):
        self.R = R if torch.is_tensor(R) else torch.Tensor(R)
        self.t = t if torch.is_tensor(t) else torch.Tensor(t)

    def transform_point(self, o):
        return torch.squeeze(self.R @ o[..., None]) + self.t

    def transform_vector(self, d):
        return torch.squeeze(self.R @ d[..., None])

    def transform_ray(self, ray):
        o = self.transform_point(ray.o)
        d = self.transform_vector(ray.d)
        dev = torch.device('cuda') if o.is_cuda else torch.device('cpu')
        return Ray(o, d, ray.wavelength, device=dev)

    def inverse(self):
        RT = self.R.T
        t = self.t
        return Transformation(RT, -RT @ t)


class Sampler(PrettyPrinter):
    def __init__(self):
        self.to()
        self.pi_over_2 = np.pi / 2
        self.pi_over_4 = np.pi / 4

    def concentric_sample_disk(self, x, y):
        x = 2 * x - 1
        y = 2 * y - 1
        cond = np.abs(x) > np.abs(y)
        r = np.where(cond, x, y)
        theta = np.where(
            cond,
            self.pi_over_4 * (y / (x + np.finfo(float).eps)),
            self.pi_over_2 - self.pi_over_4 * (x / (y + np.finfo(float).eps))
        )
        return r * np.cos(theta), r * np.sin(theta)


class Filter(PrettyPrinter):
    def __init__(self, radius):
        self.radius = radius

    def eval(self, p):
        raise NotImplementedError()


class Box(Filter):
    def __init__(self, radius=None):
        if radius is None:
            radius = [0.5, 0.5]
        Filter.__init__(self, radius)

    def eval(self, x):
        return torch.ones_like(x)


class Triangle(Filter):
    def __init__(self, radius=None):
        if radius is None:
            radius = [2.0, 2.0]
        Filter.__init__(self, radius)

    def eval(self, p):
        x, y = p[..., 0], p[..., 1]
        return (torch.maximum(torch.zeros_like(x), self.radius[0] - x) *
                torch.maximum(torch.zeros_like(y), self.radius[1] - y))


class Material(PrettyPrinter):
    def __init__(self, name=None):
        self.name = 'vacuum' if name is None else name.lower()
        self.MATERIAL_TABLE = {
            "vacuum":     [1.,       math.inf],
            "air":        [1.000293, math.inf],
            "occulder":   [1.,       math.inf],
            "f2":         [1.620,    36.37],
            "f15":        [1.60570,  37.831],
            "uvfs":       [1.458,    67.82],
            "bk10":       [1.49780,  66.954],
            "n-baf10":    [1.67003,  47.11],
            "n-bk7":      [1.51680,  64.17],
            "n-sf1":      [1.71736,  29.62],
            "n-sf2":      [1.64769,  33.82],
            "n-sf4":      [1.75513,  27.38],
            "n-sf5":      [1.67271,  32.25],
            "n-sf6":      [1.80518,  25.36],
            "n-sf6ht":    [1.80518,  25.36],
            "n-sf8":      [1.68894,  31.31],
            "n-sf10":     [1.72828,  28.53],
            "n-sf11":     [1.78472,  25.68],
            "sf1":        [1.71736,  29.51],
            "sf2":        [1.64769,  33.85],
            "sf4":        [1.75520,  27.58],
            "sf5":        [1.67270,  32.21],
            "sf6":        [1.80518,  25.43],
            "sf18":       [1.72150,  29.245],
            "baf10":      [1.67,     47.05],
            "sk16":       [1.62040,  60.306],
            "sk1":        [1.61030,  56.712],
            "ssk4":       [1.61770,  55.116],
            "b270":       [1.52290,  58.50],
            "s-nph1":     [1.8078,   22.76],
            "d-k59":      [1.5175,   63.50],
            "flint":      [1.6200,   36.37],
            "pmma":       [1.491756, 58.00],
            "polycarb":   [1.585470, 30.00],
        }
        self.A, self.B = self._lookup_material()

    def ior(self, wavelength):
        return self.A + self.B / wavelength**2

    @staticmethod
    def nV_to_AB(n, V):
        def ivs(a): return 1. / a**2
        lambdas = [656.3, 589.3, 486.1]
        B = (n - 1) / V / (ivs(lambdas[2]) - ivs(lambdas[0]))
        A = n - B * ivs(lambdas[1])
        return A, B

    def _lookup_material(self):
        out = self.MATERIAL_TABLE.get(self.name)
        if isinstance(out, list):
            n, V = out
        elif out is None:
            tmp = self.name.split('/')
            n, V = float(tmp[0]), float(tmp[1])
        return self.nV_to_AB(n, V)


class InterpolationMode(Enum):
    nearest = 1
    linear = 2


class BoundaryMode(Enum):
    zero = 1
    replicate = 2
    symmetric = 3
    periodic = 4


class SimulationMode(Enum):
    render = 1
    trace = 2


# ===========================================================================
# Utility functions (from basics.py)
# ===========================================================================

def init_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DiffMetrology is using: {}".format(device))
    torch.set_default_tensor_type('torch.FloatTensor')
    return device


def normalize(d):
    return d / torch.sqrt(torch.sum(d**2, axis=-1))[..., None]


def set_zeros(x, valid=None):
    if valid is None:
        return torch.where(torch.isnan(x), torch.zeros_like(x), x)
    else:
        mask = valid[..., None] if len(x.shape) > len(valid.shape) else valid
        return torch.where(~mask, torch.zeros_like(x), x)


def rodrigues_rotation_matrix(k, theta):
    kx, ky, kz = k[0], k[1], k[2]
    K = torch.Tensor([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ]).to(k.device)
    if not torch.is_tensor(theta):
        theta = torch.Tensor(np.asarray(theta)).to(k.device)
    return torch.eye(3, device=k.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K


def set_axes_equal(ax, scale=np.ones(3)):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    tmp = np.abs(limits[:, 1] - limits[:, 0])
    ax.set_box_aspect(scale * tmp / np.min(tmp))


# ===========================================================================
# Nested attribute access (replaces exec() calls)
# ===========================================================================

def get_nested_attr(obj, path):
    """Resolve a dot-separated attribute path, supporting array indexing like 'surfaces[0].c'."""
    parts = path.split('.')
    current = obj
    for part in parts:
        match = re.match(r'(\w+)\[(\d+)\]', part)
        if match:
            attr_name, index = match.groups()
            current = getattr(current, attr_name)[int(index)]
        else:
            current = getattr(current, part)
    return current


def set_nested_attr(obj, path, value):
    """Set attribute via dot-separated path, supporting array indexing."""
    parts = path.split('.')
    current = obj
    for part in parts[:-1]:
        match = re.match(r'(\w+)\[(\d+)\]', part)
        if match:
            attr_name, index = match.groups()
            current = getattr(current, attr_name)[int(index)]
        else:
            current = getattr(current, part)
    last = parts[-1]
    match = re.match(r'(\w+)\[(\d+)\]', last)
    if match:
        attr_name, index = match.groups()
        getattr(current, attr_name)[int(index)] = value
    else:
        setattr(current, last, value)


# ===========================================================================
# Geometric primitives (from shapes.py)
# ===========================================================================

class Endpoint(PrettyPrinter):
    def __init__(self, transformation, device=torch.device('cpu')):
        self.to_world = transformation
        self.to_object = transformation.inverse()
        self.to_world.to(device)
        self.to_object.to(device)
        self.device = device

    def intersect(self, ray):
        raise NotImplementedError()

    def sample_ray(self, position_sample=None):
        raise NotImplementedError()

    def draw_points(self, ax, options, seq=range(3)):
        raise NotImplementedError()


class Screen(Endpoint):
    def __init__(self, transformation, size, pixelsize, texture, device=torch.device('cpu')):
        self.size = torch.Tensor(np.float32(size))
        self.halfsize = self.size / 2
        self.pixelsize = torch.Tensor([pixelsize])
        self.texture = torch.Tensor(texture)
        self.texturesize = torch.Tensor(np.array(texture.shape[0:2]))
        self.texturesize_np = self.texturesize.cpu().detach().numpy()
        self.texture_shift = torch.zeros(2)
        Endpoint.__init__(self, transformation, device)
        self.to(device)

    def intersect(self, ray):
        ray_in = self.to_object.transform_ray(ray)
        t = -ray_in.o[..., 2] / ray_in.d[..., 2]
        local = ray_in(t)

        valid = (
            (t >= ray_in.mint) &
            (t <= ray_in.maxt) &
            (torch.abs(local[..., 0] - self.texture_shift[0]) <= self.halfsize[0]) &
            (torch.abs(local[..., 1] - self.texture_shift[1]) <= self.halfsize[1])
        )

        uv = (local[..., 0:2] + self.halfsize - self.texture_shift) / self.size
        uv = torch.clamp(uv, min=0.0, max=1.0)

        return local, uv, valid

    def shading(self, uv, valid, bmode=BoundaryMode.replicate, lmode=InterpolationMode.linear):
        p = uv * (self.texturesize - 1)
        p_floor = torch.floor(p).long()

        def tex(x, y):
            if bmode is BoundaryMode.replicate:
                x = torch.clamp(x, min=0, max=self.texturesize_np[0] - 1)
                y = torch.clamp(y, min=0, max=self.texturesize_np[1] - 1)
            else:
                raise NotImplementedError()
            img = self.texture[x.flatten().to(torch.int64), y.flatten().to(torch.int64)]
            return img.reshape(x.shape)

        if lmode is InterpolationMode.nearest:
            val = tex(p_floor[..., 0], p_floor[..., 1])
        elif lmode is InterpolationMode.linear:
            x0, y0 = p_floor[..., 0], p_floor[..., 1]
            s00 = tex(x0, y0)
            s01 = tex(x0, 1 + y0)
            s10 = tex(1 + x0, y0)
            s11 = tex(1 + x0, 1 + y0)
            w1 = p - p_floor
            w0 = 1. - w1
            val = (
                w0[..., 0] * (w0[..., 1] * s00 + w1[..., 1] * s01) +
                w1[..., 0] * (w0[..., 1] * s10 + w1[..., 1] * s11)
            )

        val[~valid] = 0.0
        return val

    def draw_points(self, ax, options, seq=range(3)):
        coeffs = np.array([
            [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, 1]
        ])
        points_local = torch.Tensor(coeffs * np.append(self.halfsize.cpu().detach().numpy(), 0)).to(self.device)
        points_world = self.to_world.transform_point(points_local).T.cpu().detach().numpy()
        ax.plot(points_world[seq[0]], points_world[seq[1]], points_world[seq[2]], options)


class Camera(Endpoint):
    def __init__(self, transformation,
                 filmsize, f=np.zeros(2), c=np.zeros(2), k=np.zeros(3), p=np.zeros(2),
                 device=torch.device('cpu')):
        self.filmsize = filmsize
        self.f = torch.Tensor(np.float32(f))
        self.c = torch.Tensor(np.float32(c))
        Endpoint.__init__(self, transformation, device)

        self.crop_offset = torch.zeros(2, device=device)

        self.k = torch.Tensor(np.float32(k))
        if len(self.k) < 3:
            self.k = np.append(self.k, 0)
        self.p = torch.Tensor(np.float32(p))

        self.NEWTONS_MAXITER = 5
        self.NEWTONS_TOLERANCE = 50e-6
        self.use_approximation = False
        self.to(device)

    def generate_position_sample(self, mask=None):
        dim = self.filmsize
        X, Y = torch.meshgrid(
            0.5 + dim[0] * torch.linspace(0, 1, 1 + dim[0], device=self.device)[:-1],
            0.5 + dim[1] * torch.linspace(0, 1, 1 + dim[1], device=self.device)[:-1],
        )
        if mask is not None:
            X, Y = X[mask], Y[mask]
        return torch.stack((X, Y), axis=len(X.shape))

    def sample_ray(self, position_sample=None, is_sampler=False):
        wavelength = torch.Tensor(np.asarray(562.0)).to(self.device)

        if position_sample is None:
            dim = self.filmsize
            position_sample = self.generate_position_sample()
            is_sampler = False
        else:
            dim = position_sample.shape[:-1]

        if is_sampler:
            uv = position_sample * np.float32(dim)
        else:
            uv = position_sample

        xy = self._uv2xy(uv)
        dz = torch.ones((*dim, 1), device=self.device)
        d = torch.cat((xy, dz), axis=-1)
        d = normalize(d)
        o = torch.zeros((*dim, 3), device=self.device)

        o = self.to_world.transform_point(o)
        d = self.to_world.transform_vector(d)
        ray = Ray(o, d, wavelength, self.device)
        return ray

    def draw_points(self, ax, options, seq=range(3)):
        origin = np.zeros(3)
        scales = np.append(self.filmsize / 100, 20)
        coeffs = np.array([
            [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, 1]
        ])
        sensor_corners = torch.Tensor(coeffs * scales).to(self.device)
        ps = self.to_world.transform_point(sensor_corners).T.cpu().detach().numpy()
        ax.plot(ps[seq[0]], ps[seq[1]], ps[seq[2]], options)

        for i in range(4):
            coeff = coeffs[i] * scales
            line = torch.Tensor(np.array([origin, coeff])).to(self.device)
            ps = self.to_world.transform_point(line).T.cpu().detach().numpy()
            ax.plot(ps[seq[0]], ps[seq[1]], ps[seq[2]], options)

    def _uv2xy(self, uv):
        xy_distorted = (uv + self.crop_offset - self.c) / self.f
        xy = xy_distorted
        return xy


# ===========================================================================
# Differentiable step function (from optics.py)
# ===========================================================================

class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, eps):
        ctx.constant = eps
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * torch.exp(-(ctx.constant * input)**2), None


def ind(x, eps=0.5):
    return Step.apply(x, eps)


# ===========================================================================
# Optical surfaces (from optics.py)
# ===========================================================================

class Surface(PrettyPrinter):
    def __init__(self, r, d, device=torch.device('cpu')):
        if torch.is_tensor(d):
            self.d = d
        else:
            self.d = torch.Tensor(np.asarray(float(d))).to(device)
        self.r = float(r)
        self.device = device
        self.NEWTONS_MAXITER = 10
        self.NEWTONS_TOLERANCE_TIGHT = 50e-6
        self.NEWTONS_TOLERANCE_LOOSE = 300e-6
        self.APERTURE_SAMPLING = 11

    def surface_with_offset(self, x, y):
        return self.surface(x, y) + self.d

    def normal(self, x, y):
        ds_dxyz = self.surface_derivatives(x, y)
        return normalize(torch.stack(ds_dxyz, axis=-1))

    def surface_area(self):
        return math.pi * self.r**2

    def ray_surface_intersection(self, ray, active=None):
        solution_found, local = self.newtons_method(ray.maxt, ray.o, ray.d)
        r2 = local[..., 0]**2 + local[..., 1]**2
        g = self.r**2 - r2
        if active is None:
            valid_o = solution_found & ind(g > 0.).bool()
        else:
            valid_o = active & solution_found & ind(g > 0.).bool()
        return valid_o, local, g

    def newtons_method_impl(self, maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C):
        t_delta = torch.zeros_like(oz)
        t = maxt * torch.ones_like(oz)
        residual = maxt * torch.ones_like(oz)
        it = 0
        while (torch.abs(residual) > self.NEWTONS_TOLERANCE_TIGHT).any() and (it < self.NEWTONS_MAXITER):
            it += 1
            t = t0 + t_delta
            residual, s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                t, dx, dy, dz, ox, oy, t_delta * dz, A, B, C
            )
            t_delta -= residual / s_derivatives_dot_D
        t = t0 + t_delta
        valid = (torch.abs(residual) < self.NEWTONS_TOLERANCE_LOOSE) & (t <= maxt)
        return t, t_delta, valid

    def newtons_method(self, maxt, o, D, option='implicit'):
        ox, oy, oz = (o[..., i].clone() for i in range(3))
        dx, dy, dz = (D[..., i].clone() for i in range(3))
        A = dx**2 + dy**2
        B = 2 * (dx * ox + dy * oy)
        C = ox**2 + oy**2

        t0 = (self.d - oz) / dz

        if option == 'explicit':
            t, t_delta, valid = self.newtons_method_impl(
                maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C
            )
        elif option == 'implicit':
            with torch.no_grad():
                t, t_delta, valid = self.newtons_method_impl(
                    maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C
                )
                s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                    t, dx, dy, dz, ox, oy, t_delta * dz, A, B, C
                )[1]
            t = t0 + t_delta
            t = t - (self.g(ox + t * dx, oy + t * dy) + self.h(oz + t * dz) + self.d) / s_derivatives_dot_D
        else:
            raise Exception('option={} is not available!'.format(option))

        p = o + t[..., None] * D
        return valid, p

    # Virtual methods
    def g(self, x, y):
        raise NotImplementedError()

    def dgd(self, x, y):
        raise NotImplementedError()

    def h(self, z):
        raise NotImplementedError()

    def dhd(self, z):
        raise NotImplementedError()

    def surface(self, x, y):
        raise NotImplementedError()

    def reverse(self):
        raise NotImplementedError()

    def surface_derivatives(self, x, y):
        gx, gy = self.dgd(x, y)
        z = self.surface(x, y)
        return gx, gy, self.dhd(z)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        x = ox + t * dx
        y = oy + t * dy
        s = self.g(x, y) + self.h(z)
        sx, sy = self.dgd(x, y)
        sz = self.dhd(z)
        return s, sx * dx + sy * dy + sz * dz


class Aspheric(Surface):
    def __init__(self, r, d, c=0., k=0., ai=None, device=torch.device('cpu')):
        Surface.__init__(self, r, d, device)
        self.c, self.k = (torch.Tensor(np.array(v)) for v in [c, k])
        self.ai = None
        if ai is not None:
            self.ai = torch.Tensor(np.array(ai))

    def g(self, x, y):
        return self._g(x**2 + y**2)

    def dgd(self, x, y):
        dsdr2 = 2 * self._dgd(x**2 + y**2)
        return dsdr2 * x, dsdr2 * y

    def h(self, z):
        return -z

    def dhd(self, z):
        return -torch.ones_like(z)

    def surface(self, x, y):
        return self._g(x**2 + y**2)

    def reverse(self):
        self.c = -self.c
        if self.ai is not None:
            self.ai = -self.ai

    def surface_derivatives(self, x, y):
        dsdr2 = 2 * self._dgd(x**2 + y**2)
        return dsdr2 * x, dsdr2 * y, -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        r2 = A * t**2 + B * t + C
        return self._g(r2) - z, self._dgd(r2) * (2 * A * t + B) - dz

    def _g(self, r2):
        tmp = r2 * self.c
        total_surface = tmp / (1 + torch.sqrt(1 - (1 + self.k) * tmp * self.c))
        higher_surface = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_surface = r2 * higher_surface + self.ai[i]
            higher_surface = higher_surface * r2**2
        return total_surface + higher_surface

    def _dgd(self, r2):
        alpha_r2 = (1 + self.k) * self.c**2 * r2
        tmp = torch.sqrt(1 - alpha_r2)
        total_derivative = self.c * (1 + tmp - 0.5 * alpha_r2) / (tmp * (1 + tmp)**2)
        higher_derivative = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_derivative = r2 * higher_derivative + (i + 2) * self.ai[i]
        return total_derivative + higher_derivative * r2


class BSpline(Surface):
    def __init__(self, r, d, size, px=3, py=3, tx=None, ty=None, c=None, device=torch.device('cpu')):
        Surface.__init__(self, r, d, device)
        self.px = px
        self.py = py
        self.size = np.asarray(size)

        if tx is None:
            self.tx = None
        else:
            if len(tx) != size[0] + 2 * (self.px + 1):
                raise Exception('len(tx) is not correct!')
            self.tx = torch.Tensor(np.asarray(tx)).to(self.device)
        if ty is None:
            self.ty = None
        else:
            if len(ty) != size[1] + 2 * (self.py + 1):
                raise Exception('len(ty) is not correct!')
            self.ty = torch.Tensor(np.asarray(ty)).to(self.device)

        c_shape = size + np.array([self.px, self.py]) + 1
        if c is None:
            self.c = None
        else:
            c = np.asarray(c)
            if c.size != np.prod(c_shape):
                raise Exception('len(c) is not correct!')
            self.c = torch.Tensor(c.reshape(*c_shape)).to(self.device)

        if (self.tx is None) or (self.ty is None) or (self.c is None):
            self.tx = self._generate_knots(self.r, size[0], p=px, device=device)
            self.ty = self._generate_knots(self.r, size[1], p=py, device=device)
            self.c = torch.zeros(*c_shape, device=device)
        else:
            self.to(self.device)

    @staticmethod
    def _generate_knots(R, n, p=3, device=torch.device('cpu')):
        t = np.linspace(-R, R, n)
        step = t[1] - t[0]
        T = t[0] - 0.9 * step
        np.pad(t, p + 1, 'constant', constant_values=step)
        t = np.concatenate((np.ones(p + 1) * T, t, -np.ones(p + 1) * T), axis=0)
        return torch.Tensor(t).to(device)

    def fit(self, x, y, z, eps=1e-3):
        x, y, z = (v.flatten() for v in [x, y, z])
        X = np.linspace(-self.r, self.r, self.size[0])
        Y = np.linspace(-self.r, self.r, self.size[1])
        bs = LSQBivariateSpline(x, y, z, X, Y, kx=self.px, ky=self.py, eps=eps)
        tx, ty = bs.get_knots()
        c = bs.get_coeffs().reshape(len(tx) - self.px - 1, len(ty) - self.py - 1)
        self.tx, self.ty, self.c = (torch.Tensor(v).to(self.device) for v in [tx, ty, c])

    def g(self, x, y):
        return self._deBoor2(x, y)

    def dgd(self, x, y):
        return self._deBoor2(x, y, dx=1), self._deBoor2(x, y, dy=1)

    def h(self, z):
        return -z

    def dhd(self, z):
        return -torch.ones_like(z)

    def surface(self, x, y):
        return self._deBoor2(x, y)

    def surface_derivatives(self, x, y):
        return self._deBoor2(x, y, dx=1), self._deBoor2(x, y, dy=1), -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        x = ox + t * dx
        y = oy + t * dy
        s, sx, sy = self._deBoor2(x, y, dx=-1, dy=-1)
        return s - z, sx * dx + sy * dy - dz

    def reverse(self):
        self.c = -self.c

    def _deBoor(self, x, t, c, p=3, is2Dfinal=False, dx=0):
        k = torch.sum((x[None, ...] > t[..., None]).int(), axis=0) - (p + 1)

        if is2Dfinal:
            inds = np.indices(k.shape)[0]
            def _c(jk): return c[jk, inds]
        else:
            def _c(jk): return c[jk, ...]

        need_newdim = (len(c.shape) > 1) & (not is2Dfinal)

        def f(a, b, alpha):
            if need_newdim:
                alpha = alpha[..., None]
            return (1.0 - alpha) * a + alpha * b

        if dx == 0:
            d = [_c(j + k) for j in range(0, p + 1)]
            for r in range(-p, 0):
                for j in range(p, p + r, -1):
                    left = j + k
                    t_left = t[left]
                    t_right = t[left - r]
                    alpha = (x - t_left) / (t_right - t_left)
                    d[j] = f(d[j - 1], d[j], alpha)
            return d[p]

        if dx == 1:
            q = []
            for j in range(1, p + 1):
                jk = j + k
                tmp = t[jk + p] - t[jk]
                if need_newdim:
                    tmp = tmp[..., None]
                q.append(p * (_c(jk) - _c(jk - 1)) / tmp)
            for r in range(-p, -1):
                for j in range(p - 1, p + r, -1):
                    left = j + k
                    t_right = t[left - r]
                    t_left_ = t[left + 1]
                    alpha = (x - t_left_) / (t_right - t_left_)
                    q[j] = f(q[j - 1], q[j], alpha)
            return q[p - 1]

        if dx < 0:
            d, q = [], []
            for j in range(0, p + 1):
                jk = j + k
                c_jk = _c(jk)
                d.append(c_jk)
                if j > 0:
                    tmp = t[jk + p] - t[jk]
                    if need_newdim:
                        tmp = tmp[..., None]
                    q.append(p * (c_jk - _c(jk - 1)) / tmp)
            for r in range(-p, 0):
                for j in range(p, p + r, -1):
                    left = j + k
                    t_left = t[left]
                    t_right = t[left - r]
                    alpha = (x - t_left) / (t_right - t_left)
                    d[j] = f(d[j - 1], d[j], alpha)
                    if (r < -1) & (j < p):
                        t_left_ = t[left + 1]
                        alpha = (x - t_left_) / (t_right - t_left_)
                        q[j] = f(q[j - 1], q[j], alpha)
            return d[p], q[p - 1]

    def _deBoor2(self, x, y, dx=0, dy=0):
        if not torch.is_tensor(x):
            x = torch.Tensor(np.asarray(x)).to(self.device)
        if not torch.is_tensor(y):
            y = torch.Tensor(np.asarray(y)).to(self.device)
        dim = x.shape
        x = x.flatten()
        y = y.flatten()
        x = torch.clamp(x, min=-self.r, max=self.r)
        y = torch.clamp(y, min=-self.r, max=self.r)

        if (dx == 0) & (dy == 0):
            s_tmp = self._deBoor(x, self.tx, self.c, self.px)
            s = self._deBoor(y, self.ty, s_tmp.T, self.py, True)
            return s.reshape(dim)
        elif (dx == 1) & (dy == 0):
            s_tmp = self._deBoor(y, self.ty, self.c.T, self.py)
            s_x = self._deBoor(x, self.tx, s_tmp.T, self.px, True, dx)
            return s_x.reshape(dim)
        elif (dy == 1) & (dx == 0):
            s_tmp = self._deBoor(x, self.tx, self.c, self.px)
            s_y = self._deBoor(y, self.ty, s_tmp.T, self.py, True, dy)
            return s_y.reshape(dim)
        else:
            s_tmpx = self._deBoor(x, self.tx, self.c, self.px)
            s_tmpy = self._deBoor(y, self.ty, self.c.T, self.py)
            s, s_x = self._deBoor(x, self.tx, s_tmpy.T, self.px, True, -abs(dx))
            s_y = self._deBoor(y, self.ty, s_tmpx.T, self.py, True, abs(dy))
            return s.reshape(dim), s_x.reshape(dim), s_y.reshape(dim)


class XYPolynomial(Surface):
    def __init__(self, r, d, J=0, ai=None, b=None, device=torch.device('cpu')):
        Surface.__init__(self, r, d, device)
        self.J = J
        if ai is None:
            self.ai = torch.zeros(self.J2aisize(J)) if J > 0 else torch.array([0])
        else:
            if len(ai) != self.J2aisize(J):
                raise Exception("len(ai) != (J+1)*(J+2)/2 !")
            self.ai = torch.Tensor(ai).to(device)
        if b is None:
            b = 0.
        self.b = torch.Tensor(np.asarray(b)).to(device)
        self.to(self.device)

    @staticmethod
    def J2aisize(J):
        return int((J + 1) * (J + 2) / 2)

    def center(self):
        x0 = -self.ai[2] / self.ai[5]
        y0 = -self.ai[1] / self.ai[3]
        return x0, y0

    def fit(self, x, y, z):
        x, y, z = (torch.Tensor(v.flatten()) for v in [x, y, z])
        A, AT = self._construct_A(x, y, z**2)
        coeffs = torch.solve(AT @ z[..., None], AT @ A)[0]
        self.b = coeffs[0][0]
        self.ai = coeffs[1:].flatten()

    def g(self, x, y):
        c = torch.zeros_like(x)
        count = 0
        for j in range(self.J + 1):
            for i in range(j + 1):
                c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j - i)
                count += 1
        return c

    def dgd(self, x, y):
        sx = torch.zeros_like(x)
        sy = torch.zeros_like(x)
        count = 0
        for j in range(self.J + 1):
            for i in range(j + 1):
                if j > 0:
                    sx = sx + self.ai[count] * i * torch.pow(x, max(i - 1, 0)) * torch.pow(y, j - i)
                    sy = sy + self.ai[count] * (j - i) * torch.pow(x, i) * torch.pow(y, max(j - i - 1, 0))
                count += 1
        return sx, sy

    def h(self, z):
        return self.b * z**2 - z

    def dhd(self, z):
        return 2 * self.b * z - torch.ones_like(z)

    def surface(self, x, y):
        x, y = (v if torch.is_tensor(x) else torch.Tensor(v) for v in [x, y])
        c = self.g(x, y)
        return self._solve_for_z(c)

    def reverse(self):
        self.b = -self.b
        self.ai = -self.ai

    def surface_derivatives(self, x, y):
        x, y = (v if torch.is_tensor(x) else torch.Tensor(v) for v in [x, y])
        sx = torch.zeros_like(x)
        sy = torch.zeros_like(x)
        c = torch.zeros_like(x)
        count = 0
        for j in range(self.J + 1):
            for i in range(j + 1):
                c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j - i)
                if j > 0:
                    sx = sx + self.ai[count] * i * torch.pow(x, max(i - 1, 0)) * torch.pow(y, j - i)
                    sy = sy + self.ai[count] * (j - i) * torch.pow(x, i) * torch.pow(y, max(j - i - 1, 0))
                count += 1
        z = self._solve_for_z(c)
        return sx, sy, self.dhd(z)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        x = ox + t * dx
        y = oy + t * dy
        sx = torch.zeros_like(x)
        sy = torch.zeros_like(x)
        c = torch.zeros_like(x)
        count = 0
        for j in range(self.J + 1):
            for i in range(j + 1):
                c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j - i)
                if j > 0:
                    sx = sx + self.ai[count] * i * torch.pow(x, max(i - 1, 0)) * torch.pow(y, j - i)
                    sy = sy + self.ai[count] * (j - i) * torch.pow(x, i) * torch.pow(y, max(j - i - 1, 0))
                count += 1
        s = c + self.h(z)
        return s, sx * dx + sy * dy + self.dhd(z) * dz

    def _construct_A(self, x, y, A_init=None):
        A = torch.zeros_like(x) if A_init is None else A_init
        for j in range(self.J + 1):
            for i in range(j + 1):
                A = torch.vstack((A, torch.pow(x, i) * torch.pow(y, j - i)))
        AT = A[1:, :] if A_init is None else A
        return AT.T, AT

    def _solve_for_z(self, c):
        if self.b == 0:
            return c
        else:
            return (1. - torch.sqrt(1. - 4 * self.b * c)) / (2 * self.b)


# ===========================================================================
# Lensgroup (from optics.py)
# ===========================================================================

class Lensgroup(Endpoint):
    def __init__(self, origin, shift, theta_x=0., theta_y=0., theta_z=0., device=torch.device('cpu')):
        self.origin = torch.Tensor(origin).to(device)
        self.shift = torch.Tensor(shift).to(device)
        self.theta_x = torch.Tensor(np.asarray(theta_x)).to(device)
        self.theta_y = torch.Tensor(np.asarray(theta_y)).to(device)
        self.theta_z = torch.Tensor(np.asarray(theta_z)).to(device)
        self.device = device

        Endpoint.__init__(self, self._compute_transformation(), device)

        self.mts_prepared = False

    def load_file(self, filename, lens_dir='./data/lenses/'):
        filename = filename if filename[0] == '.' else lens_dir + filename
        self.surfaces, self.materials, self.r_last, d_last = self.read_lensfile(filename)
        self.d_sensor = d_last + self.surfaces[-1].d
        self._sync()

    def load(self, surfaces, materials):
        self.surfaces = surfaces
        self.materials = materials
        self._sync()

    def _sync(self):
        for i in range(len(self.surfaces)):
            self.surfaces[i].to(self.device)

    def update(self, _x=0.0, _y=0.0):
        self.to_world = self._compute_transformation(_x, _y)
        self.to_object = self.to_world.inverse()

    def _compute_transformation(self, _x=0.0, _y=0.0, _z=0.0):
        R = (rodrigues_rotation_matrix(torch.Tensor([1, 0, 0]).to(self.device), torch.deg2rad(self.theta_x + _x)) @
             rodrigues_rotation_matrix(torch.Tensor([0, 1, 0]).to(self.device), torch.deg2rad(self.theta_y + _y)) @
             rodrigues_rotation_matrix(torch.Tensor([0, 0, 1]).to(self.device), torch.deg2rad(self.theta_z + _z)))
        t = self.origin + R @ self.shift
        return Transformation(R, t)

    @staticmethod
    def read_lensfile(filename):
        surfaces = []
        materials = []
        ds = []
        with open(filename) as file:
            line_no = 0
            d_total = 0.
            for line in file:
                if line_no < 2:
                    line_no += 1
                else:
                    ls = line.split()
                    surface_type, d, r = ls[0], float(ls[1]), float(ls[3]) / 2
                    roc = float(ls[2])
                    if roc != 0:
                        roc = 1 / roc
                    materials.append(Material(ls[4]))
                    d_total += d
                    ds.append(d)

                    if surface_type == 'O':
                        d_total = 0.
                        ds.pop()
                    elif surface_type == 'X':
                        del roc
                        ai = []
                        for ac in range(5, len(ls)):
                            if ac == 5:
                                b = float(ls[5])
                            else:
                                ai.append(float(ls[ac]))
                        surfaces.append(XYPolynomial(r, d_total, J=2, ai=ai, b=b))
                    elif surface_type == 'B':
                        del roc
                        ai = []
                        for ac in range(5, len(ls)):
                            if ac == 5:
                                nx = int(ls[5])
                            elif ac == 6:
                                ny = int(ls[6])
                            else:
                                ai.append(float(ls[ac]))
                        tx = ai[:nx + 8]
                        ai = ai[nx + 8:]
                        ty = ai[:ny + 8]
                        ai = ai[ny + 8:]
                        c = ai
                        surfaces.append(BSpline(r, d, size=[nx, ny], tx=tx, ty=ty, c=c))
                    elif surface_type == 'M':
                        raise NotImplementedError()
                    elif surface_type == 'S':
                        if len(ls) <= 5:
                            surfaces.append(Aspheric(r, d_total, roc))
                        else:
                            ai = []
                            for ac in range(5, len(ls)):
                                if ac == 5:
                                    conic = float(ls[5])
                                else:
                                    ai.append(float(ls[ac]))
                            surfaces.append(Aspheric(r, d_total, roc, conic, ai))
                    elif surface_type == 'A':
                        surfaces.append(Aspheric(r, d_total, roc))
                    elif surface_type == 'I':
                        d_total -= d
                        ds.pop()
                        materials.pop()
                        r_last = r
                        d_last = d
        return surfaces, materials, r_last, d_last

    def reverse(self):
        d_total = self.surfaces[-1].d
        for i in range(len(self.surfaces)):
            self.surfaces[i].d = d_total - self.surfaces[i].d
            self.surfaces[i].reverse()
        self.surfaces.reverse()
        self.materials.reverse()

    def rms(self, ps, units=1e3, option='centroid'):
        ps = ps[..., :2] * units
        if option == 'centroid':
            ps_mean = torch.mean(ps, axis=0)
        ps = ps - ps_mean[None, ...]
        spot_rms = torch.sqrt(torch.mean(torch.sum(ps**2, axis=-1)))
        return spot_rms

    def trace(self, ray, stop_ind=None):
        if (
            self.origin.requires_grad or self.shift.requires_grad
            or self.theta_x.requires_grad or self.theta_y.requires_grad or self.theta_z.requires_grad
        ):
            self.update()

        ray_in = self.to_object.transform_ray(ray)
        valid, mask_g, ray_out = self._trace(ray_in, stop_ind=stop_ind, record=False)
        ray_final = self.to_world.transform_ray(ray_out)
        return ray_final, valid, mask_g

    def _refract(self, wi, n, eta, approx=False):
        if np.prod(eta.shape) > 1:
            eta_ = eta[..., None]
        else:
            eta_ = eta

        cosi = torch.sum(wi * n, axis=-1)

        if approx:
            tmp = 1. - eta**2 * (1. - cosi)
            g = tmp
            valid = tmp > 0.
            wt = tmp[..., None] * n + eta_ * (wi - cosi[..., None] * n)
        else:
            cost2 = 1. - (1. - cosi**2) * eta**2
            g = cost2
            valid = cost2 > 0.
            cost2 = torch.clamp(cost2, min=1e-8)
            tmp = torch.sqrt(cost2)
            wt = tmp[..., None] * n + eta_ * (wi - cosi[..., None] * n)
        return valid, wt, g

    def _trace(self, ray, stop_ind=None, record=False):
        if stop_ind is None:
            stop_ind = len(self.surfaces) - 1
        is_forward = (ray.d[..., 2] > 0).all()

        if is_forward:
            return self._forward_tracing(ray, stop_ind, record)
        else:
            return self._backward_tracing(ray, stop_ind, record)

    def _forward_tracing(self, ray, stop_ind, record):
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape

        if record:
            oss = []
            for i in range(dim[0]):
                oss.append([ray.o[i, :].cpu().detach().numpy()])

        valid = torch.ones(dim, device=self.device).bool()
        mask = torch.ones(dim, device=self.device)
        for i in range(stop_ind + 1):
            eta = self.materials[i].ior(wavelength) / self.materials[i + 1].ior(wavelength)
            valid_o, p, g_o = self.surfaces[i].ray_surface_intersection(ray, valid)
            n = self.surfaces[i].normal(p[..., 0], p[..., 1])
            valid_d, d, g_d = self._refract(ray.d, -n, eta)
            mask = mask * ind(g_o) * ind(g_d)
            valid = valid & valid_o & valid_d
            if not valid.any():
                break
            if record:
                for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                    if v:
                        os.append(pp)
            ray.o = p
            ray.d = d

        if record:
            return valid, mask, ray, oss
        else:
            return valid, mask, ray

    def _backward_tracing(self, ray, stop_ind, record):
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape

        if record:
            oss = []
            for i in range(dim[0]):
                oss.append([ray.o[i, :].cpu().detach().numpy()])

        valid = torch.ones(dim, device=ray.o.device).bool()
        mask = torch.ones(dim, device=ray.o.device)
        for i in np.flip(range(stop_ind + 1)):
            surface = self.surfaces[i]
            eta = self.materials[i + 1].ior(wavelength) / self.materials[i].ior(wavelength)
            valid_o, p, g_o = surface.ray_surface_intersection(ray, valid)
            n = surface.normal(p[..., 0], p[..., 1])
            valid_d, d, g_d = self._refract(ray.d, n, eta)
            mask = mask * ind(g_o) * ind(g_d)
            valid = valid & valid_o & valid_d
            if not valid.any():
                break
            if record:
                for os, v, pp in zip(oss, valid.numpy(), p.cpu().detach().numpy()):
                    if v:
                        os.append(pp)
            ray.o = p
            ray.d = d

        if record:
            return valid, mask, ray, oss
        else:
            return valid, mask, ray

    def _generate_points(self, surface, with_boundary=False):
        R = surface.r
        x = y = torch.linspace(-R, R, surface.APERTURE_SAMPLING, device=self.device)
        X, Y = torch.meshgrid(x, y)
        Z = surface.surface_with_offset(X, Y)
        valid = X**2 + Y**2 <= R**2
        if with_boundary:
            from scipy import ndimage
            tmp = ndimage.convolve(valid.cpu().numpy().astype('float'), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
            boundary = valid.cpu().numpy() & (tmp != 4)
            boundary = boundary[valid.cpu().numpy()].flatten()
        points_local = torch.stack(tuple(v[valid].flatten() for v in [X, Y, Z]), axis=-1)
        points_world = self.to_world.transform_point(points_local).T.cpu().detach().numpy()
        if with_boundary:
            return points_world, boundary
        else:
            return points_world

    def draw_points(self, ax, options, seq=range(3)):
        for surface in self.surfaces:
            points_world = self._generate_points(surface)
            ax.plot(points_world[seq[0]], points_world[seq[1]], points_world[seq[2]], options)


# ===========================================================================
# Scene (from scene.py)
# ===========================================================================

class Scene(PrettyPrinter):
    def __init__(self, cameras, screen, lensgroup=None, device=torch.device('cpu')):
        self.cameras = cameras
        self.screen = screen
        self.lensgroup = lensgroup
        self.device = device
        self.camera_count = len(self.cameras)
        self.wavelength = 500

    def render(self, i=None, with_element=True, mask=None, to_numpy=False):
        im = self._simulate(i, with_element, mask, SimulationMode.render)
        if to_numpy:
            im = [x.cpu().detach().numpy() for x in im]
        return im

    def trace(self, i=None, with_element=True, mask=None, to_numpy=False):
        results = self._simulate(i, with_element, mask, SimulationMode.trace)
        p = [x[0].cpu().detach().numpy() if to_numpy else x[0] for x in results]
        valid = [x[1].cpu().detach().numpy() if to_numpy else x[1] for x in results]
        mask_g = [x[2].cpu().detach().numpy() if to_numpy else x[2] for x in results]
        return p, valid, mask_g

    def plot_setup(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if self.camera_count <= 6:
            colors = ['b', 'r', 'g', 'c', 'm', 'y']
        else:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colors = colors * np.ceil(self.camera_count / len(colors)).astype(int)
            colors = colors[:self.camera_count]

        seq = [0, 2, 1]
        self.screen.draw_points(ax, 'k', seq)

        if self.lensgroup is not None:
            self.lensgroup.draw_points(ax, 'y.', seq)

        for i, camera in enumerate(self.cameras):
            camera.draw_points(ax, colors[i], seq)

        labels = 'xyz'
        scales = np.array([2, 2, 1])
        plt.locator_params(nbins=5)
        ax.set_xlabel(labels[seq[0]] + ' [mm]')
        ax.set_ylabel(labels[seq[1]] + ' [mm]')
        ax.set_zlabel(labels[seq[2]] + ' [mm]')
        ax.legend(['Display', 'Camera 1', 'Camera 2'])
        ax.get_legend().legend_handles[0].set_color('k')
        for i in range(self.camera_count):
            ax.get_legend().legend_handles[i + 1].set_color(colors[i])
        ax.set_title('setup')
        set_axes_equal(ax, np.array([scales[seq[i]] for i in range(3)]))
        plt.show()

    def to(self, device=torch.device('cpu')):
        super().to(device)
        self.device = device
        self.lensgroup.device = device
        self.screen.device = device
        for i in range(self.camera_count):
            self.cameras[i].device = device
        for i in range(len(self.lensgroup.surfaces)):
            self.lensgroup.surfaces[i].device = device

    def _simulate(self, i=None, with_element=True, mask=None, smode=SimulationMode.render):
        def simulate(i):
            if mask is None:
                ray = self.cameras[i].sample_ray()
                if with_element and self.lensgroup is not None:
                    ray, valid_ray, mask_g = self.lensgroup.trace(ray)
                else:
                    mask_g = torch.ones(ray.o.shape[0:2], device=self.device)
                    valid_ray = mask_g.clone().bool()

                p, uv, valid_screen = self.screen.intersect(ray)
                valid = valid_screen & valid_ray

                if smode is SimulationMode.render:
                    del p
                    return self.screen.shading(uv, valid).permute(1, 0)
                elif smode is SimulationMode.trace:
                    del uv
                    return p.permute(1, 0, 2), valid.permute(1, 0), mask_g.permute(1, 0)
            else:
                mask_ = mask[i].permute(1, 0)
                ix, iy = torch.where(mask_)
                p2 = self.cameras[i].generate_position_sample(mask_)
                ray = self.cameras[i].sample_ray(p2)
                if with_element and self.lensgroup is not None:
                    ray, valid_ray, mask_g_ = self.lensgroup.trace(ray)
                else:
                    mask_g_ = torch.ones(ray.o.shape[0:2])
                    valid_ray = mask_g_.clone().bool()

                p_, uv, valid_screen = self.screen.intersect(ray)

                if smode is SimulationMode.render:
                    del p_
                    raise NotImplementedError()
                elif smode is SimulationMode.trace:
                    del uv
                    p = torch.zeros(*self.cameras[i].filmsize, 3, device=self.device)
                    p[ix, iy, ...] = p_
                    valid = torch.zeros(*self.cameras[i].filmsize, device=self.device).bool()
                    valid[ix, iy] = mask_[ix, iy]
                    mask_g = torch.zeros(*self.cameras[i].filmsize, device=self.device)
                    mask_g[ix, iy] = mask_g_
                    return p.permute(1, 0, 2), valid.permute(1, 0), mask_g.permute(1, 0)

        return [simulate(j) for j in range(self.camera_count)] if i is None else simulate(i)


# ===========================================================================
# High-level convenience functions
# ===========================================================================

def set_texture(scene, texture, device):
    """Set the screen display pattern from a texture image."""
    if len(texture.shape) > 2:
        texture = texture[0]
    pixelsize = scene.screen.pixelsize.item()
    sizenew = pixelsize * np.array(texture.shape)
    t = np.zeros(3)
    scene.screen = Screen(Transformation(np.eye(3), t), sizenew, pixelsize, np.flip(texture).copy(), device)
