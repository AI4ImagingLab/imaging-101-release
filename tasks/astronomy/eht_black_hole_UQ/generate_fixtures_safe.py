#!/usr/bin/env python
"""
Generate test fixtures for eht_black_hole_UQ WITHOUT using NFFTInfo (which hangs).

Replaces compute_nufft_params's NFFTInfo call with a direct analytical
computation of the pulse function correction factor.
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIXTURE_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation", "fixtures")

os.makedirs(os.path.join(FIXTURE_BASE, "preprocessing"), exist_ok=True)
os.makedirs(os.path.join(FIXTURE_BASE, "physics_model"), exist_ok=True)
os.makedirs(os.path.join(FIXTURE_BASE, "visualization"), exist_ok=True)


def compute_nufft_params_safe(obs, npix, fov_uas):
    """Compute NUFFT params without NFFTInfo (avoids pynfft hang)."""
    import ehtim as eh
    import ehtim.const_def as ehc

    fov = fov_uas * eh.RADPERUAS
    psize = fov / npix

    obs_data = obs.unpack(['u', 'v'])
    uv = np.hstack((obs_data['u'].reshape(-1, 1), obs_data['v'].reshape(-1, 1)))
    vu = np.hstack((obs_data['v'].reshape(-1, 1), obs_data['u'].reshape(-1, 1)))

    # Compute pulsefac analytically for trianglePulse2D:
    # FT of triangle pulse = sinc^2(u*psize) * sinc^2(v*psize)
    pulsefac = np.sinc(uv[:, 0] * psize) ** 2 * np.sinc(uv[:, 1] * psize) ** 2

    # Scale trajectory for torchkbnufft: vu * psize * 2π
    vu_scaled = np.array(vu * psize * 2 * np.pi)
    ktraj_vis = torch.tensor(vu_scaled.T, dtype=torch.float32).unsqueeze(0)
    pulsefac_vis_torch = torch.tensor(
        np.concatenate([np.expand_dims(pulsefac.real, 0),
                        np.expand_dims(pulsefac.imag, 0)], 0),
        dtype=torch.float32
    )

    return {
        "ktraj_vis": ktraj_vis,
        "pulsefac_vis": pulsefac_vis_torch,
    }


def generate_preprocessing_fixtures():
    print("Generating preprocessing fixtures...")
    import ehtim as eh
    from src.preprocessing import (
        load_observation, load_metadata, extract_closure_indices,
        build_prior_image
    )

    # load_observation
    obs_data = load_observation("data")
    np.savez(os.path.join(FIXTURE_BASE, "preprocessing", "load_observation.npz"),
             output_vis_shape=np.array(obs_data['vis'].shape),
             output_uv_shape=np.array(obs_data['uv_coords'].shape),
             output_vis_first5=obs_data['vis'][:5])

    # extract_closure_indices
    obs = obs_data['obs']
    closure = extract_closure_indices(obs)
    closure_fixture = {
        'output_n_cphase': np.array(len(closure['cphase_ind_list'][0])),
        'output_n_camp': np.array(len(closure['camp_ind_list'][0])),
    }
    for i in range(3):
        closure_fixture[f'output_cphase_ind{i}'] = closure['cphase_ind_list'][i]
        closure_fixture[f'output_cphase_sign{i}'] = closure['cphase_sign_list'][i]
    for i in range(4):
        closure_fixture[f'output_camp_ind{i}'] = closure['camp_ind_list'][i]
    np.savez(os.path.join(FIXTURE_BASE, "preprocessing", "extract_closure_indices.npz"),
             **closure_fixture)

    # compute_nufft_params — use SAFE version
    nufft = compute_nufft_params_safe(obs, 32, 160.0)
    np.savez(os.path.join(FIXTURE_BASE, "preprocessing", "compute_nufft_params.npz"),
             output_ktraj_shape=np.array(nufft['ktraj_vis'].shape),
             output_pulsefac_shape=np.array(nufft['pulsefac_vis'].shape),
             output_ktraj_vis=nufft['ktraj_vis'].numpy(),
             output_pulsefac_vis=nufft['pulsefac_vis'].numpy())

    # build_prior_image
    obs2 = eh.obsdata.load_uvfits(os.path.join("data", "obs.uvfits"))
    prior, flux_const = build_prior_image(obs2, 32, 160.0)
    np.savez(os.path.join(FIXTURE_BASE, "preprocessing", "build_prior_image.npz"),
             output_prior=prior,
             output_flux_const=np.array(flux_const))

    print("  Done.")
    return obs_data, closure, nufft, prior, flux_const


def generate_physics_model_fixtures(obs_data, closure, nufft):
    print("Generating physics_model fixtures...")
    from src.physics_model import (
        NUFFTForwardModel,
        Loss_angle_diff, Loss_logca_diff2, Loss_l1, Loss_TSV, Loss_flux, Loss_center,
    )

    device = torch.device("cpu")

    # Build model
    model = NUFFTForwardModel(
        32, nufft['ktraj_vis'], nufft['pulsefac_vis'],
        [torch.tensor(a, dtype=torch.long) for a in closure['cphase_ind_list']],
        [torch.tensor(a, dtype=torch.float32) for a in closure['cphase_sign_list']],
        [torch.tensor(a, dtype=torch.long) for a in closure['camp_ind_list']],
        device
    )

    # Test image: simple Gaussian
    np.random.seed(42)
    test_img = np.abs(np.random.randn(2, 32, 32)).astype(np.float32)
    images = torch.tensor(test_img)

    vis, visamp, cphase, logcamp = model(images)

    np.savez(os.path.join(FIXTURE_BASE, "physics_model", "nufft_forward.npz"),
             input_images=test_img,
             output_vis=vis.detach().numpy(),
             output_vis_shape=np.array(vis.shape),
             output_visamp_shape=np.array(visamp.shape),
             output_cphase=cphase.detach().numpy(),
             output_cphase_shape=np.array(cphase.shape),
             output_logcamp=logcamp.detach().numpy(),
             output_logcamp_shape=np.array(logcamp.shape))

    # Loss function fixtures
    obs = obs_data['obs']

    # Loss_angle_diff
    sigma_cp = np.array(closure['cphase_data']['sigmacp'][:10], dtype=np.float32)
    true_cp = np.random.randn(2, 10).astype(np.float32) * 30
    pred_cp = true_cp + np.random.randn(2, 10).astype(np.float32) * 5
    loss_fn = Loss_angle_diff(sigma_cp, device)
    loss_val = loss_fn(torch.tensor(true_cp), torch.tensor(pred_cp))
    np.savez(os.path.join(FIXTURE_BASE, "physics_model", "loss_angle_diff.npz"),
             input_sigma=sigma_cp, input_true=true_cp, input_pred=pred_cp,
             output_loss=loss_val.detach().numpy())

    # Loss_logca_diff2
    sigma_ca = np.abs(np.random.randn(10).astype(np.float32)) + 0.1
    true_ca = np.random.randn(2, 10).astype(np.float32)
    pred_ca = true_ca + np.random.randn(2, 10).astype(np.float32) * 0.1
    loss_fn2 = Loss_logca_diff2(sigma_ca, device)
    loss_val2 = loss_fn2(torch.tensor(true_ca), torch.tensor(pred_ca))
    np.savez(os.path.join(FIXTURE_BASE, "physics_model", "loss_logca_diff2.npz"),
             input_sigma=sigma_ca, input_true=true_ca, input_pred=pred_ca,
             output_loss=loss_val2.detach().numpy())

    # Image prior losses
    test_img_batch = torch.tensor(test_img)  # (2, 32, 32)
    l1_val = Loss_l1(test_img_batch)
    tsv_val = Loss_TSV(test_img_batch)
    flux_val = Loss_flux(1.0)(test_img_batch)
    center_val = Loss_center(device, center=15.5, dim=32)(test_img_batch)

    np.savez(os.path.join(FIXTURE_BASE, "physics_model", "loss_priors.npz"),
             input_image=test_img,
             output_l1=l1_val.detach().numpy(),
             output_tsv=tsv_val.detach().numpy(),
             config_flux=np.array(1.0),
             output_flux=flux_val.detach().numpy(),
             output_center=center_val.detach().numpy())

    print("  Done.")


def generate_visualization_fixtures():
    print("Generating visualization fixtures...")
    from src.visualization import compute_metrics, compute_uq_metrics

    np.random.seed(42)
    gt = np.abs(np.random.randn(32, 32)).astype(np.float64) + 0.01
    estimate = gt + np.random.randn(32, 32) * 0.1

    metrics = compute_metrics(estimate, gt)
    np.savez(os.path.join(FIXTURE_BASE, "visualization", "compute_metrics.npz"),
             input_estimate=estimate,
             input_ground_truth=gt,
             output_nrmse=np.array(metrics['nrmse']),
             output_ncc=np.array(metrics['ncc']),
             output_dynamic_range=np.array(metrics['dynamic_range']))

    # UQ metrics
    std = np.abs(np.random.randn(32, 32)) * 0.2 + 0.05
    uq_metrics = compute_uq_metrics(estimate, std, gt)
    np.savez(os.path.join(FIXTURE_BASE, "visualization", "compute_uq_metrics.npz"),
             input_mean=estimate,
             input_std=std,
             input_gt=gt,
             output_calibration=np.array(uq_metrics['calibration']),
             output_mean_uncertainty=np.array(uq_metrics['mean_uncertainty']))

    print("  Done.")


if __name__ == "__main__":
    obs_data, closure, nufft, prior, flux_const = generate_preprocessing_fixtures()
    generate_physics_model_fixtures(obs_data, closure, nufft)
    generate_visualization_fixtures()
    print("\nAll fixtures generated successfully!")
