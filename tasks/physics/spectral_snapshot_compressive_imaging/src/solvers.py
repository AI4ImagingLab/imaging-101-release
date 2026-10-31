"""Inverse solvers: GAP and ADMM with plug-and-play deep denoiser priors."""

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from skimage.restoration import denoise_tv_chambolle

from .physics_model import A, At, shift, shift_back


# ---------------------------------------------------------------------------
# Neural network denoiser (HSI-SDeCNN)
# ---------------------------------------------------------------------------

def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)
    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)


def _sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def _conv_relu(in_channels=128, out_channels=128, kernel_size=3, stride=1,
               padding=1, bias=True, if_relu=True):
    L = []
    L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    if if_relu:
        L.append(nn.ReLU(inplace=True))
    return _sequential(*L)


class HSI_SDeCNN(nn.Module):
    """Hyperspectral image denoiser based on SDeCNN architecture.

    Parameters
    ----------
    in_nc : int
        Number of input channels (spectral window size).
    out_nc : int
        Number of output channels.
    nc : int
        Number of intermediate feature channels.
    nb : int
        Number of convolutional layers.
    """
    def __init__(self, in_nc=7, out_nc=1, nc=128, nb=15):
        super(HSI_SDeCNN, self).__init__()
        sf = 2
        self.m_down = PixelUnShuffle(upscale_factor=sf)
        m_head = _conv_relu(in_nc * sf * sf + 1, nc)
        m_body = [_conv_relu(nc, nc) for _ in range(nb - 2)]
        m_tail = _conv_relu(nc, out_nc * sf * sf, if_relu=False)
        self.model = _sequential(m_head, *m_body, m_tail)
        self.m_up = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x, sigma):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 2) * 2 - h)
        paddingRight = int(np.ceil(w / 2) * 2 - w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        x = self.m_down(x)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)
        x = x[..., :h, :w]
        return x


def load_denoiser(checkpoint_path, device=None):
    """Load pretrained HSI_SDeCNN denoiser.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pth checkpoint file.
    device : torch.device, optional
        Device to load the model on.

    Returns
    -------
    model : HSI_SDeCNN
        Loaded model in eval mode.
    device : torch.device
        Device the model is on.
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = HSI_SDeCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    for q, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model, device


# ---------------------------------------------------------------------------
# TV denoiser
# ---------------------------------------------------------------------------

def TV_denoiser(x, _lambda, n_iter_max):
    """Total variation denoiser using gradient descent.

    Parameters
    ----------
    x : ndarray, shape (H, W, nC)
        Noisy input.
    _lambda : float
        Regularization weight.
    n_iter_max : int
        Number of iterations.

    Returns
    -------
    u : ndarray
        Denoised output.
    """
    dt = 0.25
    N = x.shape
    idx = np.arange(1, N[0] + 1)
    idx[-1] = N[0] - 1
    iux = np.arange(-1, N[0] - 1)
    iux[0] = 0
    ir = np.arange(1, N[1] + 1)
    ir[-1] = N[1] - 1
    il = np.arange(-1, N[1] - 1)
    il[0] = 0
    p1 = np.zeros_like(x)
    p2 = np.zeros_like(x)
    divp = np.zeros_like(x)

    for i in range(n_iter_max):
        z = divp - x * _lambda
        z1 = z[:, ir, :] - z
        z2 = z[idx, :, :] - z
        denom_2d = 1 + dt * np.sqrt(np.sum(z1 ** 2 + z2 ** 2, 2))
        denom_3d = np.tile(denom_2d[:, :, np.newaxis], (1, 1, N[2]))
        p1 = (p1 + dt * z1) / denom_3d
        p2 = (p2 + dt * z2) / denom_3d
        divp = p1 - p1[:, il, :] + p2 - p2[iux, :, :]
    u = x - divp / _lambda
    return u


# ---------------------------------------------------------------------------
# CNN band-by-band denoiser helper
# ---------------------------------------------------------------------------

def _denoise_cnn_bands(x, model, device, nC, noise_levels):
    """Apply the CNN denoiser band-by-band with spectral context window.

    Parameters
    ----------
    x : ndarray, shape (H, W, nC)
        Input spectral cube.
    model : HSI_SDeCNN
        Pretrained denoiser.
    device : torch.device
    nC : int
        Number of spectral channels.
    noise_levels : tuple of float
        (low_ch, mid_ch, high_ch) noise levels for boundary/interior bands.

    Returns
    -------
    tem : ndarray, shape (H, W, nC)
        Denoised spectral cube.
    """
    l_ch, m_ch, h_ch = noise_levels
    tem = None
    for i in range(nC):
        if i < 3:
            if i == 0:
                net_input = np.dstack((x[:, :, i], x[:, :, i], x[:, :, i], x[:, :, i:i + 4]))
            elif i == 1:
                net_input = np.dstack((x[:, :, i - 1], x[:, :, i - 1], x[:, :, i - 1], x[:, :, i:i + 4]))
            elif i == 2:
                net_input = np.dstack((x[:, :, i - 2], x[:, :, i - 2], x[:, :, i - 1], x[:, :, i:i + 4]))
            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0, 1).float().unsqueeze(0)
            net_input = net_input.to(device)
            Nsigma = torch.full((1, 1, 1, 1), l_ch / 255.).type_as(net_input)
            output = model(net_input, Nsigma)
            output = output.data.squeeze().cpu().numpy()
        elif i > nC - 4:
            if i == nC - 3:
                net_input = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i + 1], x[:, :, i + 2], x[:, :, i + 2]))
            elif i == nC - 2:
                net_input = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i + 1], x[:, :, i + 1], x[:, :, i + 1]))
            elif i == nC - 1:
                net_input = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i], x[:, :, i], x[:, :, i]))
            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0, 1).float().unsqueeze(0)
            net_input = net_input.to(device)
            Nsigma = torch.full((1, 1, 1, 1), m_ch / 255.).type_as(net_input)
            output = model(net_input, Nsigma)
            output = output.data.squeeze().cpu().numpy()
        else:
            net_input = x[:, :, i - 3:i + 4]
            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0, 1).float().unsqueeze(0)
            net_input = net_input.to(device)
            Nsigma = torch.full((1, 1, 1, 1), h_ch / 255.).type_as(net_input)
            output = model(net_input, Nsigma)
            output = output.data.squeeze().cpu().numpy()

        if i == 0:
            tem = output
        else:
            tem = np.dstack((tem, output))
    return tem


# ---------------------------------------------------------------------------
# GAP solver
# ---------------------------------------------------------------------------

def _use_cnn(k):
    """Determine whether to use CNN denoiser at iteration k."""
    return ((k >= 122 and k <= 125) or (k >= 119 and k <= 121) or
            (k >= 115 and k <= 117) or (k >= 111 and k <= 113) or
            (k >= 107 and k <= 109) or (k >= 103 and k <= 105) or
            (k >= 99 and k <= 101) or (k >= 95 and k <= 97) or
            (k >= 91 and k <= 93) or (k >= 87 and k <= 89) or
            (k >= 83 and k <= 85))


def gap_denoise(y, Phi, _lambda=1, accelerate=True, iter_max=20,
                sigma=None, tv_iter_max=5, x0=None, X_orig=None,
                checkpoint_path=None, show_iqa=True):
    """GAP-based reconstruction with plug-and-play deep denoiser prior.

    Parameters
    ----------
    y : ndarray, shape (H, W)
        Compressed 2D measurement.
    Phi : ndarray, shape (H, W, nC)
        3D sensing matrix.
    _lambda : float
        Regularization factor.
    accelerate : bool
        Enable accelerated GAP.
    iter_max : int
        Maximum iterations per sigma level.
    sigma : list of float
        Noise levels for each stage.
    tv_iter_max : int
        TV denoising iterations.
    x0 : ndarray, optional
        Initial estimate.
    X_orig : ndarray, optional
        Ground truth for computing metrics.
    checkpoint_path : str
        Path to denoiser checkpoint.
    show_iqa : bool
        Whether to compute and print metrics.

    Returns
    -------
    x : ndarray
        Reconstructed spectral cube (shifted).
    psnr_all : list of float
        PSNR at each iteration.
    ssim_all : list of float
        SSIM at each iteration.
    """
    from .visualization import psnr, calculate_ssim

    if x0 is None:
        x0 = At(y, Phi)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)

    y1 = np.zeros_like(y)
    Phi_sum = np.sum(Phi, 2)
    Phi_sum[Phi_sum == 0] = 1

    x = x0
    psnr_all = []
    ssim_all = []
    k = 0

    model, device = load_denoiser(checkpoint_path)

    done = False
    for idx, nsig in enumerate(sigma):
        if done:
            break
        for it in range(iter_max[idx]):
            yb = A(x, Phi)
            if accelerate:
                y1 = y1 + (y - yb)
                x = x + _lambda * (At((y1 - yb) / Phi_sum, Phi))
            else:
                x = x + _lambda * (At((y - yb) / Phi_sum, Phi))

            x = shift_back(x, step=1)

            if _use_cnn(k):
                x = _denoise_cnn_bands(x, model, device, x.shape[2], (10, 10, 10))
            else:
                x = denoise_tv_chambolle(x, nsig / 255, max_num_iter=tv_iter_max,
                                         channel_axis=-1)

            if show_iqa and X_orig is not None:
                ssim_all.append(calculate_ssim(X_orig, x))
                psnr_all.append(psnr(X_orig, x))
                print('  GAP-HSICNN iteration {: 3d}, '
                      'PSNR {:2.2f} dB, SSIM {:.4f}'.format(
                          k + 1, psnr_all[-1], ssim_all[-1]))

            x = shift(x, step=1)
            if k == 123:
                done = True
                break
            k = k + 1

    return x, psnr_all, ssim_all


# ---------------------------------------------------------------------------
# ADMM solver
# ---------------------------------------------------------------------------

def admm_denoise(y, Phi, _lambda=1, gamma=0.01, iter_max=50,
                 sigma=None, tv_weight=0.1, tv_iter_max=5, x0=None,
                 X_orig=None, checkpoint_path=None, show_iqa=True):
    """ADMM-based reconstruction with plug-and-play deep denoiser prior.

    Parameters
    ----------
    y : ndarray, shape (H, W)
        Compressed 2D measurement.
    Phi : ndarray, shape (H, W, nC)
        3D sensing matrix.
    _lambda : float
        Regularization factor.
    gamma : float
        ADMM penalty parameter.
    iter_max : int
        Maximum iterations per sigma level.
    sigma : list of float
        Noise levels for each stage.
    tv_weight : float
        TV denoising weight.
    tv_iter_max : int
        TV denoising iterations.
    x0 : ndarray, optional
        Initial estimate.
    X_orig : ndarray, optional
        Ground truth for computing metrics.
    checkpoint_path : str
        Path to denoiser checkpoint.
    show_iqa : bool
        Whether to compute and print metrics.

    Returns
    -------
    theta : ndarray
        Reconstructed spectral cube (shifted).
    psnr_all : list of float
    ssim_all : list of float
    """
    from .visualization import psnr, calculate_ssim

    if x0 is None:
        x0 = At(y, Phi)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)

    x = x0
    theta = x0
    Phi_sum = np.sum(Phi, 2)
    Phi_sum[Phi_sum == 0] = 1
    b = np.zeros_like(x0)
    psnr_all = []
    ssim_all = []
    k = 0

    model, device = load_denoiser(checkpoint_path)

    for idx, nsig in enumerate(sigma):
        for it in range(iter_max[idx]):
            yb = A(theta + b, Phi)
            x = (theta + b) + _lambda * (At((y - yb) / (Phi_sum + gamma), Phi))
            x1 = shift_back(x - b, step=2)

            if k >= 89:
                theta = _denoise_cnn_bands(x1, model, device, x1.shape[2], (10, 10, 10))
            else:
                theta = TV_denoiser(x1, tv_weight, n_iter_max=tv_iter_max)

            if show_iqa and X_orig is not None:
                psnr_all.append(psnr(X_orig, theta))
                ssim_all.append(calculate_ssim(X_orig, theta))
                print('  ADMM-HSICNN iteration {: 3d}, '
                      'PSNR {:2.2f} dB, SSIM {:.4f}'.format(
                          k + 1, psnr_all[-1], ssim_all[-1]))

            theta = shift(theta, step=2)
            b = b - (x - theta)
            k = k + 1

    return theta, psnr_all, ssim_all
