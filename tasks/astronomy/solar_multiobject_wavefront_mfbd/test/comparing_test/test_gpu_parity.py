"""Device-specific original full-batch derivatives at two continuation stages."""
from pathlib import Path
import numpy as np
import pytest
import torch
from src.preprocessing import (load_data, normalize_frames, extract_patches,
    estimate_noise, cosine_window, apodize)
from src.physics_model import build_optics, canonical_modes, pupil_to_psf, reduced_loss

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Original CUDA arithmetic parity requires a CUDA device')
def test_original_full_batch_cuda_gradient_and_mask():
    observations, mixing = load_data(ROOT/'data/raw_data.npz')
    patches = extract_patches(normalize_frames(observations))
    window = cosine_window().cuda()
    frames = [torch.fft.fft2(apodize(x.cuda(), window).cpu().contiguous(), norm='ortho').cuda()
              for x in patches]
    sigmas = [estimate_noise(x).cuda() for x in patches]
    optics = build_optics(mixing, device='cuda')
    with np.load(ROOT/'test/fixtures/upstream_gpu_stages.npz', allow_pickle=False) as upstream:
        np.testing.assert_array_equal(optics['diffraction_mask'].cpu(), upstream['diffraction_mask'])
        for channel in range(2):
            np.testing.assert_array_equal(sigmas[channel].cpu(), upstream['noise'][channel])
        for index in range(2):
            modes = torch.tensor(upstream['modes'][index], device='cuda', requires_grad=True)
            active = int(upstream['active'][index])
            centered = canonical_modes(modes)
            otfs = [pupil_to_psf(centered, optics['pupil'], optics['basis'], active)[1] for _ in range(2)]
            loss = reduced_loss(frames, otfs, sigmas, optics['diffraction_mask'])
            np.testing.assert_allclose(float(loss.detach()), upstream['losses'][index], rtol=2e-6)
            loss.backward()
            np.testing.assert_allclose(modes.grad.cpu(), upstream['gradients'][index], rtol=3e-5, atol=2e-8)
