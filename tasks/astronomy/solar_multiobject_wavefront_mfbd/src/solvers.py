"""Explicit staged wavefront optimization and analytic multi-object projection."""
import time
import numpy as np
import torch
from .preprocessing import (normalize_frames, extract_patches, estimate_noise,
                            cosine_window, apodize, mosaic)
from .physics_model import (build_optics, canonical_modes, pupil_to_psf,
                            reduced_loss, project_objects)


def annealing_schedule(iterations=250):
    """Return cumulative radial-degree activation, as updated after each Adam step."""
    levels = np.cumsum(np.arange(2, 11))
    result = np.zeros(iterations, dtype=np.int64)
    for iteration in range(iterations):
        if iteration < 0.1*iterations:
            result[iteration] = levels[0]
        elif iteration > 0.6*iterations:
            result[iteration] = levels[-1]
        else:
            index = np.clip(8/(0.5*iterations)*(iteration-0.1*iterations), 0, 8)
            result[iteration] = levels[int(index)]
    return result


def optimize_wavefronts(frame_fts, sigmas, optics, iterations=250, progress=None):
    """Optimize all patch/frame KL coefficients with Adam and the reduced loss."""
    device = frame_fts[0].device
    modes = torch.zeros((*frame_fts[0].shape[:2], 54), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([{'params': modes, 'lr': 0.08}])
    # The source constructs a cosine scheduler but never steps it: LR remains 0.08.
    active = 2
    schedule = annealing_schedule(iterations)
    losses = torch.zeros(iterations, device=device)
    for iteration in range(iterations):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            centered = canonical_modes(modes)
            otfs = [pupil_to_psf(centered, optics['pupil'], optics['basis'], active)[1]
                    for _ in range(2)]
            loss = reduced_loss(frame_fts, otfs, sigmas, optics['diffraction_mask'])
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        losses[iteration] = loss.detach()
        if progress is not None:
            progress(iteration, float(loss.detach()), active)
        active = int(schedule[iteration])
    return canonical_modes(modes.detach()), losses.detach()


def reconstruct(observations, mixing, device=None, iterations=250, progress=None):
    """Run the complete 121-patch, two-object, twelve-frame reconstruction."""
    started = time.perf_counter()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patches = extract_patches(normalize_frames(observations))
    sigmas = [estimate_noise(patches[i]).to(device) for i in range(2)]
    window = cosine_window().to(device)
    # Original apodization follows the window device, then copies to a CPU FFT buffer.
    prepared = [apodize(patches[i].to(device), window).cpu().contiguous() for i in range(2)]
    frame_fts = [torch.fft.fft2(prepared[i], norm='ortho').to(device) for i in range(2)]
    optics = build_optics(mixing, device=device)
    modes, losses = optimize_wavefronts(frame_fts, sigmas, optics, iterations, progress)
    with torch.no_grad():
        otfs = [pupil_to_psf(modes, optics['pupil'], optics['basis'])[1] for _ in range(2)]
        objects = project_objects(frame_fts, otfs, sigmas, optics['diffraction_mask'])
        reconstruction = mosaic(objects)
    return {'reconstruction': reconstruction.cpu().numpy()[None],
            'objects': objects.cpu().numpy()[None],
            'wavefront_coefficients': modes.cpu().numpy()[None],
            'loss_history': losses.cpu().numpy(),
            'runtime_seconds': time.perf_counter()-started}
