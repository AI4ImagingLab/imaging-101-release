"""Expected values were frozen by an independent upstream CPU execution."""
import numpy as np
import torch
from src.preprocessing import load_data, normalize_frames, extract_patches, estimate_noise, apodize, cosine_window
from src.physics_model import canonical_modes, pupil_to_psf, reduced_loss, project_objects
from src.solvers import optimize_wavefronts, annealing_schedule
from conftest import ROOT


def inputs(oracle):
    return ([torch.tensor(oracle[f'output_frame_ft{i}']) for i in range(2)],
            [torch.tensor(oracle[f'output_sigma{i}']) for i in range(2)])


def test_bound_preprocessing_and_noise(oracle):
    observations, _ = load_data(ROOT/'data/raw_data.npz')
    patches = extract_patches(normalize_frames(observations))
    assert len(patches) == 2 and patches[0].shape == (121, 12, 88, 88)
    for channel in range(2):
        np.testing.assert_array_equal(patches[channel][:1], oracle[f'input_patches{channel}'])
        np.testing.assert_allclose(estimate_noise(patches[channel][:1]),
                                   oracle[f'output_sigma{channel}'], rtol=1e-6, atol=1e-8)
        prepared = apodize(patches[channel][:1], cosine_window())
        np.testing.assert_array_equal(prepared, oracle[f'output_apodized{channel}'])
        np.testing.assert_allclose(torch.fft.fft2(prepared, norm='ortho'),
                                   oracle[f'output_frame_ft{channel}'], rtol=2e-6, atol=2e-6)


def test_optics_extraction(oracle, optics):
    for local, upstream in [('basis', 'basis'), ('pupil', 'pupil'),
                             ('frequency_radius', 'frequency_radius'),
                             ('diffraction_mask', 'diffraction_mask')]:
        np.testing.assert_array_equal(optics[local], oracle[f'output_{upstream}'])
    np.testing.assert_array_equal(cosine_window(), oracle['output_window'])


def test_full_mode_psf_loss_gradient(oracle, optics):
    modes = torch.tensor(oracle['input_test_modes'], requires_grad=True)
    psfs, otfs = [], []
    for channel in range(2):
        psf, otf = pupil_to_psf(canonical_modes(modes), optics['pupil'], optics['basis'])
        np.testing.assert_allclose(psf.detach(), oracle[f'output_test_psf{channel}'], rtol=2e-6, atol=1e-8)
        np.testing.assert_allclose(otf.detach(), oracle[f'output_test_otf{channel}'], rtol=3e-6, atol=1e-9)
        psfs.append(psf)
        otfs.append(otf)
    frames, sigmas = inputs(oracle)
    loss = reduced_loss(frames, otfs, sigmas, optics['diffraction_mask'])
    np.testing.assert_allclose(loss.detach(), oracle['output_test_loss'], rtol=2e-6)
    loss.backward()
    np.testing.assert_allclose(modes.grad, oracle['output_test_gradient'], rtol=3e-5, atol=2e-6)


def test_first_adam_step_and_final_object(oracle, optics):
    frames, sigmas = inputs(oracle)
    modes, losses = optimize_wavefronts(frames, sigmas, optics, iterations=1)
    np.testing.assert_allclose(modes, canonical_modes(torch.tensor(oracle['output_modes'])), rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(losses, oracle['output_loss'], rtol=2e-6)
    np.testing.assert_array_equal(annealing_schedule(1), oracle['output_annealing'])
    otfs = [pupil_to_psf(modes, optics['pupil'], optics['basis'])[1] for _ in range(2)]
    objects = project_objects(frames, otfs, sigmas, optics['diffraction_mask'])
    for channel in range(2):
        np.testing.assert_allclose(objects[channel], oracle[f'output_object{channel}'], rtol=2e-6, atol=2e-6)
