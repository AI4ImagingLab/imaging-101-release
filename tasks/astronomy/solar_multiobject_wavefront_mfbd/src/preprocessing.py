"""Observation preparation extracted from torchmfbd; legal text in test/validation.md."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as functional

IMAGE_SIZE = 512
PATCH_SIZE = 88
STRIDE = 40
APODIZATION = 6


def load_data(path='data/raw_data.npz'):
    """Return observed intensities (2,12,512,512) and theoretical KL mixing table."""
    with np.load(Path(path), allow_pickle=False) as data:
        observations = data['observations']
        mixing = data['kl_mixing']
    if observations.shape != (1, 2, 12, 512, 512) or observations.dtype != np.float32:
        raise ValueError('observations must be float32 (1,2,12,512,512)')
    if mixing.shape != (1, 436, 3) or mixing.dtype != np.float32:
        raise ValueError('kl_mixing must be float32 (1,436,3)')
    if not np.isfinite(observations).all() or not np.isfinite(mixing).all():
        raise ValueError('Inputs must be finite')
    return observations[0], mixing[0]


def normalize_frames(observations):
    """Normalize each complete sensor frame by its own float32 spatial mean."""
    frames = np.array(observations, dtype=np.float32, copy=True)
    means = np.mean(frames, axis=(-1, -2), keepdims=True)
    if np.any(means <= 0):
        raise ValueError('Frame means must be positive')
    frames /= means
    return torch.tensor(frames)


def extract_patches(frames, patch_size=PATCH_SIZE, stride=STRIDE):
    """Return two row-major (patches,frames,rows,columns) strided tensors."""
    result = []
    for channel in frames:
        unfolded = functional.unfold(channel[None], kernel_size=patch_size, stride=stride)
        count = unfolded.shape[-1]
        result.append(unfolded.reshape(channel.shape[0], patch_size, patch_size, count)
                      .permute(3, 0, 1, 2))
    return result


def cosine_window(size=PATCH_SIZE, border=APODIZATION):
    """Separable Hann-tapered window with an unattenuated interior."""
    edge = np.hanning(2 * border)
    line = np.ones(size)
    line[:border] = edge[:border]
    line[-border:] = edge[-border:]
    return torch.tensor(np.outer(line, line).astype('float32'))


def apodize(patches, window):
    """Suppress boundary variation around each patch-frame's spatial mean."""
    mean = torch.mean(patches, dim=(-1, -2), keepdim=True)
    return (patches - mean) * window + mean


def estimate_noise(patches):
    """Gavish-Donoho estimate from singular values of each spatial covariance."""
    values = patches.detach().cpu().numpy()
    rows, columns = values.shape[-2:]
    beta = rows / columns
    correction = 8.0 * beta / (beta + 1 + np.sqrt(beta**2 + 14*beta + 1))
    lam = np.sqrt(2.0 * (beta + 1) + correction)
    omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
    covariance = values @ np.swapaxes(values, -2, -1)
    _, singular, _ = np.linalg.svd(covariance)
    sigma = np.median(np.sqrt(singular), axis=-1) / (np.sqrt(columns) * lam / omega)
    return torch.tensor(sigma.astype('float32'))


def mosaic(patch_objects, apodization=APODIZATION, weight_border=30, stride=STRIDE):
    """Cosine overlap-add of 121 reconstructed patches, returning (objects,476,476)."""
    objects, count, size, _ = patch_objects.shape
    side = int(np.sqrt(count))
    if side * side != count:
        raise ValueError('Expected a square patch grid')
    interior = patch_objects[..., apodization:-apodization, apodization:-apodization]
    kernel = interior.shape[-1]
    weight = cosine_window(kernel, min(weight_border, kernel//2)).to(patch_objects.device)
    output_size = (side-1)*stride + kernel
    result = torch.zeros((objects, output_size, output_size), dtype=patch_objects.dtype,
                         device=patch_objects.device)
    normalizer = torch.zeros_like(result)
    for index in range(count):
        row, column = divmod(index, side)
        region = (slice(None), slice(row*stride, row*stride+kernel),
                  slice(column*stride, column*stride+kernel))
        result[region] += interior[:, index] * weight
        normalizer[region] += weight
    return result / (normalizer + 1e-10)
