"""Explicit rotate/sum single-photon acquisition mathematics.

Adapted from PyTomography (MIT), and homogeneous rotation conventions from
Kornia (Apache-2.0). See NOTICE and test/validation.md. Inverse-angle bilinear
backprojection is the published continuum approximation, not the exact
transpose of discrete interpolation. All axes are (x,y,z), lengths in cm.
"""
import numpy as np
import torch
import torch.nn.functional as F

DX = 0.47951998710631993
DELTA = 1e-11
PSF_PARAMETERS = (0.03161982752665455, 0.1248503046423388, 0.16137114205472364)


def rotation_grid(angle, height, width, device, dtype):
    """Centred inverse-coordinate bilinear sampling, align_corners=True.

    Preserve source float32 pi and pixel-normalization constants even when
    testing float64 images; the matrix multiplications preserve source order.
    """
    center = torch.tensor([[(width-1)/2, (height-1)/2]], device=device, dtype=dtype)
    theta = torch.tensor([float(angle)], device=device, dtype=dtype)
    theta = theta * torch.tensor(np.pi, dtype=torch.float32, device=device).to(dtype) / 180
    c, s = torch.cos(theta), torch.sin(theta)
    shift = torch.eye(3, dtype=dtype, device=device)[None].clone()
    shift[:, :2, 2] = center
    inverse_shift = torch.eye(3, dtype=dtype, device=device)[None].clone()
    inverse_shift[:, :2, 2] = -center
    rot = torch.eye(3, dtype=dtype, device=device)[None].clone()
    rot[:, 0, 0], rot[:, 0, 1] = c, s
    rot[:, 1, 0], rot[:, 1, 1] = -s, c
    scale = torch.eye(3, dtype=dtype, device=device)[None]
    matrix = shift @ rot @ scale @ inverse_shift
    normal = torch.tensor([[[1.,0.,-1.],[0.,1.,-1.],[0.,0.,1.]]])
    normal[:,0,0] = normal[:,0,0] * 2 / (width-1)
    normal[:,1,1] = normal[:,1,1] * 2 / (height-1)
    normal = normal.to(device=device, dtype=dtype)
    normalized = normal @ (matrix @ torch.linalg.inv(normal))
    return F.affine_grid(torch.inverse(normalized)[:,:2,:], [1,1,height,width], align_corners=True)


def rotate_volume(volume, grid):
    return F.grid_sample(volume.permute(2,0,1)[None], grid, mode='bilinear',
                         padding_mode='zeros', align_corners=True)[0].permute(1,2,0)


def detection_probability(mu, dx=DX):
    """Half starting voxel plus complete downstream voxels, detector at +x."""
    a = mu * dx
    integral = torch.flip(torch.cumsum(torch.flip(a, [0]), dim=0), [0]) - a/2
    return torch.exp(-integral)


def gaussian_kernel(radius, size, dx=DX, device='cpu'):
    distance = np.maximum(radius + ((size-1)/2-np.arange(size))*dx, 0)
    a,b,c = PSF_PARAMETERS
    sigmas = np.sqrt((a*distance+b)**2+c**2)/dx
    width = int(np.ceil(max(sigmas)*3)*2+1)
    x = torch.arange(-(width//2), width//2+1, device=device)[None,None].repeat(size,1,1)
    sigma = torch.tensor(sigmas, dtype=torch.float32, device=device).reshape(size,1,1)
    kernel = torch.exp(-x**2/(2*sigma**2+DELTA))
    return kernel/kernel.sum(axis=-1).unsqueeze(-1)


def blur_volume(volume, kernel):
    """Separable detector-parallel y,z Gaussian; zero boundary extension."""
    groups = volume.shape[0]
    out = F.conv1d(volume.permute(2,0,1), kernel, groups=groups, padding='same')
    out = F.conv1d(out.permute(2,1,0), kernel, groups=groups, padding='same')
    out = out.permute(2,1,0)
    return out.permute(1,2,0)


class Acquisition:
    """Known geometry/attenuation/PSF; explicit full and subset FP/BP."""
    def __init__(self, attenuation, angles, radii, device='cpu', attenuation_enabled=True,
                 psf_enabled=True, n_subsets=8):
        self.device = torch.device(device)
        self.mu = torch.as_tensor(attenuation, dtype=torch.float32, device=device)
        self.shape = tuple(self.mu.shape)
        self.pad = int(np.ceil((np.sqrt(2)*self.shape[1]-self.shape[1])/2))
        self.padded_shape = (self.shape[0]+2*self.pad,self.shape[1]+2*self.pad,self.shape[2])
        self.mu_padded = F.pad(self.mu,[0,0,self.pad,self.pad,self.pad,self.pad])
        self.angles = torch.as_tensor(angles, dtype=torch.float32, device=device)
        self.phis = (270-self.angles).tolist()
        self.radii = np.asarray(radii)
        self.n_subsets = n_subsets
        self.attenuation_enabled = attenuation_enabled
        self.psf_enabled = psf_enabled
        self.grids = {}
        self.kernels = {float(r): gaussian_kernel(r,self.padded_shape[0],device=device) for r in np.unique(radii)}
        self.boundary = torch.zeros(self.padded_shape, device=device)
        self.boundary[:,self.pad:-self.pad,:] = 1

    def indices(self, subset=None):
        return list(range(len(self.phis))) if subset is None else list(range(subset,len(self.phis),self.n_subsets))

    def grid(self, angle):
        key = float(angle)
        if key not in self.grids:
            self.grids[key] = rotation_grid(key,*self.padded_shape[:2],self.device,torch.float32)
        return self.grids[key]

    def probability(self, index):
        return detection_probability(rotate_volume(self.mu_padded,self.grid(-self.phis[index])))

    @torch.no_grad()
    def forward(self, volume, subset=None):
        padded = F.pad(volume.to(self.device), [0,0,self.pad,self.pad,self.pad,self.pad])
        indices = self.indices(subset)
        proj = torch.zeros((len(indices),self.padded_shape[1],self.shape[2]),device=self.device)
        for j,i in enumerate(indices):
            u = rotate_volume(padded,self.grid(-self.phis[i]))
            if self.attenuation_enabled: u = u*self.probability(i)
            if self.psf_enabled: u = blur_volume(u,self.kernels[float(self.radii[i])])
            proj[j] = u.sum(axis=0)
        return proj[:,self.pad:-self.pad,:]

    @torch.no_grad()
    def backward(self, projections, subset=None):
        p = F.pad(projections.to(self.device),[0,0,self.pad,self.pad])
        result = torch.zeros(self.padded_shape,device=self.device)
        for j,i in enumerate(self.indices(subset)):
            u = p[j].unsqueeze(0)*self.boundary
            if self.psf_enabled: u = blur_volume(u,self.kernels[float(self.radii[i])])
            if self.attenuation_enabled: u = u*self.probability(i)
            result += rotate_volume(u,self.grid(self.phis[i]))
        return result[self.pad:-self.pad,self.pad:-self.pad,:]
