"""Explicit ER/HIO, shrink-wrap, coherence estimation and ensemble averaging.

Algorithm adapted from cohere, Copyright 2019 UChicago Argonne, LLC. Full
license and the independently justified coherence correction are in validation.
No upstream/domain reconstruction package is imported here.
"""
import time
import numpy as np
from scipy import fft
from . import physics_model as model

STARTS = 200
ITERATIONS = 1020
SELECTED = 10
KERNEL_SHAPE = (16, 16, 16)
SIGMAS = (3., 2.5, 2., 1.5, 1.)


class ArrayBackend:
    """Numerical primitives only; task-specific updates remain below."""
    def __init__(self, device="cpu"):
        self.device = device
        self.gpu = device != "cpu"
        self.cupy = device == "cupy"
        if self.cupy:
            import cupy
            import cupyx.scipy.fft as cufft
            import cupyx.scipy.signal as cusignal
            import cupyx.scipy.ndimage as cundimage
            self.xp = cupy
            cupy.get_default_memory_pool().set_limit(size=3 * 1024**3)
            self.cufft = cufft
            self.cusignal = cusignal
            self.cundimage = cundimage
        elif self.gpu:
            import torch
            self.torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.backends.cudnn.benchmark = False
            self.xp = torch
        else:
            self.xp = np

    def array(self, value):
        if self.cupy:
            return self.xp.asarray(value)
        return self.torch.as_tensor(value, device=self.device) if self.gpu else np.asarray(value)

    def numpy(self, value):
        if self.cupy:
            return self.xp.asnumpy(value)
        return value.detach().cpu().numpy() if self.gpu else np.asarray(value)

    def fft(self, value, inverse=False):
        if self.cupy:
            operation = self.cufft.ifftn if inverse else self.cufft.fftn
            return self.cufft.fftshift(operation(self.cufft.ifftshift(value), norm="forward"))
        if self.gpu:
            operation = self.torch.fft.ifftn if inverse else self.torch.fft.fftn
            return self.torch.fft.fftshift(operation(
                self.torch.fft.ifftshift(value), norm="forward"))
        return model.reciprocal_wavefield(value) if inverse else model.direct_wavefield(value)

    def full_convolution(self, array, kernel):
        if self.cupy:
            return self.cusignal.fftconvolve(array, kernel, mode="full")
        if not self.gpu:
            return model.full_convolution(array, kernel)
        shape = tuple(a + b - 1 for a, b in zip(array.shape, kernel.shape))
        padded = tuple(fft.next_fast_len(n, real=True) for n in shape)
        tfft = self.torch.fft
        full = tfft.irfftn(tfft.rfftn(array, s=padded) * tfft.rfftn(kernel, s=padded), s=padded)
        return full[tuple(slice(0, n) for n in shape)]

    def convolution(self, array, kernel):
        full = self.full_convolution(array, kernel)
        start = tuple((n - 1) // 2 for n in kernel.shape)
        return full[tuple(slice(s, s + n) for s, n in zip(start, array.shape))]

    def kernel_adjoint(self, residual, coherent, kernel_shape):
        origin = tuple((n - 1) // 2 for n in kernel_shape)
        start = tuple(n - 1 - o for n, o in zip(coherent.shape, origin))
        full = self.full_convolution(residual, self.flip(coherent))
        return full[tuple(slice(s, s + n) for s, n in zip(start, kernel_shape))]

    def gaussian(self, array, sigma):
        if self.cupy:
            return self.cundimage.gaussian_filter(array, sigma)
        if not self.gpu:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(array, sigma)
        torch = self.torch
        radius = int(4 * sigma + .5)
        coordinate = torch.arange(-radius, radius + 1, dtype=array.dtype, device=self.device)
        weights = torch.exp(-.5 * (coordinate / sigma) ** 2)
        weights /= weights.sum()
        result = array
        # Explicit half-sample reflection also works when radius exceeds an axis.
        for axis in range(3):
            n = result.shape[axis]
            indices = torch.arange(-radius, n + radius, device=self.device) % (2 * n)
            indices = torch.where(indices < n, indices, 2 * n - 1 - indices)
            padded = torch.index_select(result, axis, indices)
            shape = [1, 1, 1]
            shape[axis] = weights.numel()
            result = torch.nn.functional.conv3d(
                padded[None, None], weights.reshape(1, 1, *shape))[0, 0]
        return result

    def flip(self, value):
        if self.cupy:
            return self.xp.flip(value)
        return self.torch.flip(value, tuple(range(value.ndim))) if self.gpu else np.flip(value)

    def square_root(self, value):
        if self.cupy:
            return self.xp.sqrt(self.xp.maximum(value, 0.))
        return self.xp.sqrt(self.xp.clamp(value, min=0.)) if self.gpu else np.sqrt(np.maximum(value, 0.))


def initial_object(shape, amplitude_max, seed):
    """Uniform complex64 random start, centered half-box support, upstream scale."""
    rng = np.random.default_rng(seed)
    image = (rng.random(shape, dtype=np.float32) + 1j *
             rng.random(shape, dtype=np.float32)).astype(np.complex64)
    image *= np.sum(np.abs(image) ** 2, dtype=np.float32) * np.float32(amplitude_max)
    support = np.zeros(shape, dtype=np.float64)
    size = tuple(int(n * .5) for n in shape)
    support[tuple(slice((a - b) // 2, (a - b) // 2 + b) for a, b in zip(shape, size))] = 1.
    image *= support
    return image, support


def reconstruction_step(image, support, amplitude, hio=False, beta=.9, kernel=None):
    """CPU ER/HIO single step; fixture-tested independently of ensemble driver."""
    wavefield = model.reciprocal_wavefield(image)
    projected, error = model.amplitude_projection(wavefield, amplitude, kernel)
    projected = model.direct_wavefield(projected)
    update = np.where(support > 0, projected, image - beta * projected) if hio else projected * support
    return update, error


def reconstruct_one(amplitude, seed=0, device="cpu", initial=None, iterations=ITERATIONS,
                    progress=None, stage_callback=None):
    """Run one selected 1020-step protocol; return object, kernel, error history.

    Shorter iterations are allowed for unit diagnostics only. Standard CLI and
    notebook always use all 1020 and all 200 starts. Initial is a hidden test
    injection for upstream parity, never a public observation or cached result.
    """
    if not 1 <= iterations <= ITERATIONS:
        raise ValueError("iterations must be between 1 and 1020")
    backend = ArrayBackend(device)
    xp = backend.xp
    image, support = initial_object(amplitude.shape, float(amplitude.max()), seed)
    if initial is not None:
        if initial.shape != amplitude.shape:
            raise ValueError("initial shape differs from observations")
        image = np.asarray(initial).copy()
    image, support = backend.array(image), backend.array(support)
    measured = backend.array(amplitude)
    measured_roi = model.crop_center(measured, KERNEL_SHAPE)
    # The original GPU full() wrapper ignores its dtype keyword: actual float64.
    kernel = backend.array(np.full(KERNEL_SHAPE, .5, dtype=np.float64))
    previous = None
    errors = []
    for step in range(iterations):
        cycle = min(step // 200, 4)
        local_step = step % 200
        if local_step >= 10 and local_step != 20 and (local_step - 10) % 10 == 0:
            smoothed = backend.gaussian(xp.abs(image), SIGMAS[cycle])
            support = xp.where(smoothed >= .1 * smoothed.max(), 1, 0)
        wavefield = backend.fft(image, inverse=True)
        absolute = xp.abs(wavefield)
        if step >= 400 and (step - 400) % 50 == 0:
            roi = model.crop_center(absolute, KERNEL_SHAPE)
            coherent = xp.square(2 * roi - previous)
            normalization = xp.square(measured_roi).sum() / coherent.sum()
            # Preserve the upstream amplitude normalization and its rounding.
            coherent = xp.square(xp.sqrt(coherent * normalization))
            sensitivity = backend.kernel_adjoint(
                xp.ones_like(measured_roi, dtype=xp.float64), coherent, KERNEL_SHAPE)
            sensitivity = xp.where(sensitivity > 0., sensitivity, 1.)
            for _ in range(20):
                denominator = backend.convolution(coherent, kernel)
                denominator = xp.where(denominator > 1e-30, denominator, 1e-30)
                kernel = kernel * backend.kernel_adjoint(
                    xp.square(measured_roi) / denominator, coherent, KERNEL_SHAPE) / sensitivity
                kernel = xp.where(kernel > 0., kernel, 0.)
            kernel = xp.abs(kernel.real)
            kernel = kernel / kernel.sum()
        converged = absolute if step < 400 else backend.square_root(
            backend.convolution(xp.square(absolute), kernel))
        denominator = xp.where(converged > 1e-9, converged, 1.)
        ratio = xp.where(converged > 1e-9, measured / denominator, 0.)
        residual = xp.where(converged != 0., converged - measured, 0.)
        error = xp.square(xp.abs(residual)).sum() / xp.square(xp.abs(measured)).sum()
        projected_wave = wavefield * ratio
        if step >= 399 and (step - 399) % 50 == 0:
            previous = model.crop_center(xp.abs(projected_wave), KERNEL_SHAPE).copy() \
                if not backend.gpu or backend.cupy else model.crop_center(
                    xp.abs(projected_wave), KERNEL_SHAPE).clone()
        projected = backend.fft(projected_wave)
        is_hio = step < 1000 and local_step >= 20
        image = xp.where(support > 0, projected, image - .9 * projected) if is_hio else projected * support
        errors.append(float(error))
        if stage_callback is not None:
            stage_callback(step + 1, image, support, kernel, errors[-1], backend)
        if progress is not None and step % 100 == 0:
            progress(step, errors[-1])
    image = image / xp.abs(image).max()
    return backend.numpy(image), backend.numpy(kernel), np.asarray(errors)


def align_complex(reference, candidate):
    """Choose direct/conjugate-inverted twin and integer cyclic shift, then phase.

    This exact complex cross-correlation convention is used for ensemble
    averaging. The choice is calculated from estimated objects, never truth.
    """
    best = None
    for twin in (False, True):
        other = np.conj(np.flip(candidate)) if twin else candidate
        correlation = fft.ifftn(fft.fftn(reference) * np.conj(fft.fftn(other)))
        index = np.unravel_index(np.argmax(np.abs(correlation)), reference.shape)
        value = correlation[index]
        score = float(np.abs(value))
        if best is None or score > best[0]:
            shifted = np.roll(other, index, axis=tuple(range(other.ndim)))
            phase = value / abs(value) if value != 0 else 1.
            best = (score, shifted * phase, twin, index)
    return best[1], {"twin": best[2], "shift": [int(i) for i in best[3]]}


def aggregate_ensemble(objects, kernels, amplitude):
    """Align selected estimates, average complex object/kernel, restore count scale."""
    result = objects[0].copy()
    alignment = [{"twin": False, "shift": [0, 0, 0]}]
    for obj in objects[1:]:
        value, parameters = align_complex(objects[0], obj)
        result += value
        alignment.append(parameters)
    result /= len(objects)
    coherence = np.mean(kernels, axis=0)
    coherence /= coherence.sum()
    prediction = np.sqrt(model.observation_intensity(result, coherence))
    scale = np.sum(amplitude * prediction) / np.sum(prediction ** 2)
    result *= scale
    return result, coherence, alignment


class StoredObjects:
    """Read only this run's selected objects, one candidate at a time."""

    def __init__(self, paths):
        self.paths = paths
        self.anchor = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return (np.load(path, allow_pickle=False) for path in self.paths[index])
        if index == 0:
            if self.anchor is None:
                self.anchor = np.load(self.paths[0], allow_pickle=False)
            return self.anchor
        return np.load(self.paths[index], allow_pickle=False)


def reconstruct(counts, starts=STARTS, device="cupy", progress=None):
    """Fresh full two-position reconstruction; no checkpoint/result cache is read."""
    if starts < SELECTED:
        raise ValueError("at least ten starts are required for the defined ensemble")
    from .preprocessing import prepare_observations
    from pathlib import Path
    from tempfile import TemporaryDirectory
    output, coherences, histories = [], [], []
    with TemporaryDirectory(prefix="bragg-fresh-selected-") as scratch:
        for position, observed in enumerate(counts):
            amplitude, preprocessing = prepare_observations(observed)
            peak_shift = preprocessing["peak_shift"]
            best = []
            errors = []
            for start in range(starts):
                seed = 10000 + position * 1000 + start
                begin = time.perf_counter()
                obj, kernel, history = reconstruct_one(amplitude, seed=seed, device=device)
                errors.append(float(history[-1]))
                ranking = sorted(best + [(errors[-1], start, None, kernel)],
                                 key=lambda item: (item[0], item[1]))[:SELECTED]
                if any(item[1] == start for item in ranking):
                    path = Path(scratch)/f"p{position}-start{start:03}.npy"
                    np.save(path, obj, allow_pickle=False)
                    ranking = [(error, index, path if index == start else old_path, value)
                               for error, index, old_path, value in ranking]
                retained = {item[1] for item in ranking}
                for _, index, old_path, _ in best:
                    if index not in retained:
                        old_path.unlink()
                best = ranking
                del obj
                if progress is not None:
                    progress(position, start, errors[-1], time.perf_counter() - begin)
            if device == "cupy":
                # All start-local GPU arrays are dead before CPU-only aggregation.
                import cupy as cp
                cp.fft.config.get_plan_cache().clear()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            obj, kernel, alignment = aggregate_ensemble(
                StoredObjects([b[2] for b in best]), [b[3] for b in best], amplitude)
            output.append(obj)
            coherences.append(kernel)
            histories.append({"errors": errors, "selected": [b[1] for b in best],
                              "alignment": alignment, "peak_shift": peak_shift.tolist(),
                              "intensity_threshold": preprocessing["intensity_threshold"],
                              "padding": preprocessing["padding"]})
    return np.stack(output)[None], np.stack(coherences)[None], histories
