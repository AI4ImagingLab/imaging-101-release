#!/usr/bin/env python
"""Generate test fixtures for fpm_inr_reconstruction.

Creates fixtures in evaluation/fixtures/ subdirectories.
CUDA-dependent fixtures are created only if CUDA is available.
"""

import os
import sys
import json
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

DATA_DIR = os.path.join(TASK_DIR, "data")
FIXTURES_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")

PREPROCESSING_DIR = os.path.join(FIXTURES_DIR, "preprocessing")
PHYSICS_DIR = os.path.join(FIXTURES_DIR, "physics_model")
SOLVERS_DIR = os.path.join(FIXTURES_DIR, "solvers")
NETWORK_DIR = os.path.join(FIXTURES_DIR, "network")
VIS_DIR = os.path.join(FIXTURES_DIR, "visualization")
AIF_DIR = os.path.join(FIXTURES_DIR, "all_in_focus")

for d in [PREPROCESSING_DIR, PHYSICS_DIR, SOLVERS_DIR, NETWORK_DIR, VIS_DIR, AIF_DIR]:
    os.makedirs(d, exist_ok=True)


def generate_preprocessing_fixtures():
    """Generate preprocessing fixtures (CPU only).

    Returns None-tuple if raw_data.npz is not available.
    """
    from src.preprocessing import load_metadata

    print("  Generating preprocessing fixtures...")
    metadata = load_metadata(DATA_DIR)

    raw_data_path = os.path.join(DATA_DIR, "raw_data.npz")
    if not os.path.exists(raw_data_path):
        print("    [SKIP] raw_data.npz not found - skipping data-dependent preprocessing fixtures")
        print("    (Only metadata-based fixtures will be generated)")
        return metadata, None, None, None, None

    from src.preprocessing import (
        load_raw_data, compute_optical_params,
        compute_pupil_and_propagation, compute_z_params,
        load_ground_truth,
    )

    raw_data = load_raw_data(DATA_DIR)
    optical_params = compute_optical_params(raw_data, metadata)
    pupil_data = compute_pupil_and_propagation(optical_params)
    z_params = compute_z_params(metadata, optical_params)

    # --- load_raw_data ---
    np.savez(os.path.join(PREPROCESSING_DIR, "load_raw_data.npz"),
             output_I_low_shape=np.array(raw_data["I_low"].shape),
             output_na_calib=raw_data["na_calib"],
             output_mag=raw_data["mag"],
             output_dpix_c=raw_data["dpix_c"],
             output_na_cal=raw_data["na_cal"])
    print("    [OK] preprocessing/load_raw_data.npz")

    # --- compute_optical_params ---
    np.savez(os.path.join(PREPROCESSING_DIR, "compute_optical_params.npz"),
             output_Fxx1=optical_params["Fxx1"],
             output_Fyy1=optical_params["Fyy1"],
             output_ledpos_true=optical_params["ledpos_true"],
             output_order=optical_params["order"],
             output_M=optical_params["M"],
             output_N=optical_params["N"],
             output_MM=optical_params["MM"],
             output_NN=optical_params["NN"],
             output_k0=optical_params["k0"],
             output_kmax=optical_params["kmax"],
             output_Isum_shape=np.array(optical_params["Isum"].shape),
             output_Isum_max=optical_params["Isum"].max())
    print("    [OK] preprocessing/compute_optical_params.npz")

    # --- load_ground_truth ---
    gt_path = os.path.join(DATA_DIR, "ground_truth.npz")
    if os.path.exists(gt_path):
        gt = load_ground_truth(DATA_DIR)
        np.savez(os.path.join(PREPROCESSING_DIR, "load_ground_truth.npz"),
                 output_I_stack_shape=np.array(gt["I_stack"].shape),
                 output_zvec=gt["zvec"])
        print("    [OK] preprocessing/load_ground_truth.npz")
    else:
        print("    [SKIP] ground_truth.npz not found")

    # --- compute_pupil ---
    np.savez(os.path.join(PREPROCESSING_DIR, "compute_pupil.npz"),
             output_Pupil0=pupil_data["Pupil0"],
             output_Pupil0_sum=pupil_data["Pupil0"].sum(),
             output_kzz_real=pupil_data["kzz"].real.astype("float32"),
             output_kzz_imag=pupil_data["kzz"].imag.astype("float32"))
    print("    [OK] preprocessing/compute_pupil.npz")

    # --- compute_z_params ---
    with open(os.path.join(PREPROCESSING_DIR, "compute_z_params.json"), "w") as f:
        json.dump(z_params, f, indent=2)
    print("    [OK] preprocessing/compute_z_params.json")

    return metadata, raw_data, optical_params, pupil_data, z_params


def generate_visualization_fixtures():
    """Generate visualization / all-in-focus fixtures (CPU only)."""
    from src.visualization import (
        create_balance_map,
        all_in_focus_normal_variance,
        compute_ssim_per_slice,
        compute_allfocus_l2,
    )

    print("  Generating visualization fixtures...")

    # --- create_balance_map ---
    n_patches, bmap = create_balance_map(256, patch_size=64, patch_pace=16)
    np.savez(os.path.join(AIF_DIR, "create_balance_map.npz"),
             input_image_size=256,
             input_patch_size=64,
             input_patch_pace=16,
             output_n_patches=n_patches,
             output_balance_map=bmap)
    print("    [OK] all_in_focus/create_balance_map.npz")

    # --- all_in_focus ---
    rng = np.random.default_rng(42)
    z_stack = rng.random((256, 256, 5)).astype("float32") * 0.5 + 0.3
    # Make one slice sharper
    z_stack[100:156, 100:156, 2] = 0.9
    aif = all_in_focus_normal_variance(z_stack, patch_size=64, patch_pace=16)
    np.savez(os.path.join(AIF_DIR, "all_in_focus.npz"),
             input_z_stack=z_stack,
             output_aif=aif)
    print("    [OK] all_in_focus/all_in_focus.npz")

    # --- compute_ssim_per_slice ---
    rng2 = np.random.default_rng(7)
    pred_ssim = rng2.random((5, 64, 64)).astype("float32")
    gt_ssim = pred_ssim + rng2.normal(0, 0.05, pred_ssim.shape).astype("float32")
    gt_ssim = np.clip(gt_ssim, 0, 1)
    from skimage.metrics import structural_similarity as ssim
    ssim_vals = np.array([
        ssim(pred_ssim[i], gt_ssim[i], data_range=1.0) for i in range(5)
    ])
    np.savez(os.path.join(VIS_DIR, "compute_ssim_per_slice.npz"),
             input_pred=pred_ssim,
             input_gt=gt_ssim,
             output_ssim=ssim_vals)
    print("    [OK] visualization/compute_ssim_per_slice.npz")

    # --- compute_allfocus_l2 ---
    rng3 = np.random.default_rng(12)
    aif_pred = rng3.random((64, 64)).astype("float32")
    aif_gt = aif_pred + rng3.normal(0, 0.02, aif_pred.shape).astype("float32")
    pred_c = aif_pred - np.mean(aif_pred)
    gt_c = aif_gt - np.mean(aif_gt)
    mse = float(np.mean((pred_c - gt_c) ** 2))
    psnr = float(10 * np.log10(1.0 / (mse + 1e-10)))
    np.savez(os.path.join(VIS_DIR, "compute_allfocus_l2.npz"),
             input_pred=aif_pred,
             input_gt=aif_gt,
             output_mse=mse,
             output_psnr=psnr)
    print("    [OK] visualization/compute_allfocus_l2.npz")


def generate_cuda_fixtures(metadata, optical_params, pupil_data, z_params):
    """Generate CUDA-dependent fixtures. Skipped if no GPU available."""
    import torch
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available - skipping physics_model, network, solver fixtures")
        return

    from src.physics_model import FPMForwardModel
    from src.solvers import FullModel, load_model_with_required_grad, FPMINRSolver, G_Renderer
    from src.preprocessing import prepare_data

    device = "cuda:0"
    print("  Generating CUDA fixtures...")

    Pupil0 = (
        torch.from_numpy(pupil_data["Pupil0"])
        .view(1, 1, optical_params["M"], optical_params["N"])
        .to(device)
    )
    kzz = torch.from_numpy(pupil_data["kzz"]).to(device).unsqueeze(0)
    Isum_data = prepare_data(DATA_DIR, device=device)
    Isum_tensor = Isum_data["Isum"]

    fwd = FPMForwardModel(
        Pupil0=Pupil0, kzz=kzz,
        ledpos_true=optical_params["ledpos_true"],
        M=optical_params["M"], N=optical_params["N"],
        MAGimg=optical_params["MAGimg"],
    )

    # --- prepare_data fixture ---
    np.savez(os.path.join(PREPROCESSING_DIR, "prepare_data.npz"),
             output_Isum_shape=np.array(list(Isum_tensor.shape)))
    print("    [OK] preprocessing/prepare_data.npz")

    # --- compute_spectrum_mask ---
    dz = torch.tensor([0.0]).to(device)
    spec_mask = fwd.compute_spectrum_mask(dz, [0])
    np.savez(os.path.join(PHYSICS_DIR, "compute_spectrum_mask.npz"),
             output_shape=np.array(list(spec_mask.shape)),
             output_spectrum_mask_real=spec_mask.cpu().numpy().real.astype("float32"),
             output_spectrum_mask_imag=spec_mask.cpu().numpy().imag.astype("float32"))
    print("    [OK] physics_model/compute_spectrum_mask.npz")

    # --- get_led_coords ---
    led_num = [0, 1, 2]
    x_0, y_0, x_1, y_1 = fwd.get_led_coords(led_num)
    np.savez(os.path.join(PHYSICS_DIR, "get_led_coords.npz"),
             input_led_num=np.array(led_num),
             output_x_0=np.array(x_0),
             output_y_0=np.array(y_0),
             output_x_1=np.array(x_1),
             output_y_1=np.array(y_1))
    print("    [OK] physics_model/get_led_coords.npz")

    # --- get_measured_amplitudes ---
    n_z = 1
    meas_amp = fwd.get_measured_amplitudes(Isum_tensor, led_num, n_z)
    np.savez(os.path.join(PHYSICS_DIR, "get_measured_amplitudes.npz"),
             input_led_num=np.array(led_num),
             input_n_z=n_z,
             output_shape=np.array(list(meas_amp.shape)),
             output_crop=meas_amp[0, 0, :64, :64].float().cpu().numpy())
    print("    [OK] physics_model/get_measured_amplitudes.npz")

    # --- Load trained model for network fixtures ---
    weights_path = os.path.join(TASK_DIR, "evaluation", "reference_outputs", "model_weights.pth")
    if os.path.exists(weights_path):
        model = FullModel(
            w=optical_params["MM"], h=optical_params["MM"],
            num_feats=32, x_mode=metadata["num_modes"], y_mode=metadata["num_modes"],
            z_min=-20.0, z_max=20.0, ds_factor=1, use_layernorm=False,
        ).to(device)
        load_model_with_required_grad(model, weights_path)
        model.eval()

        with torch.no_grad():
            ampli, phase = model(torch.tensor([0.0]).to(device))

        crop = slice(512, 576)
        np.savez(os.path.join(NETWORK_DIR, "full_model_forward.npz"),
                 output_amplitude_shape=np.array(list(ampli.shape)),
                 output_amplitude_crop=ampli[0, crop, crop].float().cpu().numpy(),
                 output_phase_crop=phase[0, crop, crop].float().cpu().numpy(),
                 output_amplitude_mean=ampli.float().mean().cpu().item(),
                 output_phase_mean=phase.float().mean().cpu().item())
        print("    [OK] network/full_model_forward.npz")

        # --- save_load_model fixture (just save an empty marker) ---
        np.savez(os.path.join(NETWORK_DIR, "save_load_model.npz"),
                 placeholder=np.array([0]))
        print("    [OK] network/save_load_model.npz")

        # --- get_sub_spectrum ---
        spec_mask2 = fwd.compute_spectrum_mask(dz, led_num)
        with torch.no_grad():
            ampli2, phase2 = model(dz)
            img_complex = ampli2 * torch.exp(1j * phase2)
        sub_spec = fwd.get_sub_spectrum(img_complex, led_num, spec_mask2)
        np.savez(os.path.join(PHYSICS_DIR, "get_sub_spectrum.npz"),
                 input_led_num=np.array(led_num),
                 output_shape=np.array(list(sub_spec.shape)),
                 output_crop=sub_spec[0, 0, :64, :64].float().cpu().numpy())
        print("    [OK] physics_model/get_sub_spectrum.npz")

        # --- evaluate fixture ---
        solver = FPMINRSolver()
        z_positions = np.array([0.0, 5.0, -5.0])
        ampli_eval, phase_eval = solver.evaluate(model, z_positions, device=device)
        np.savez(os.path.join(SOLVERS_DIR, "evaluate.npz"),
                 input_z_positions=z_positions,
                 output_amplitude_shape=np.array(ampli_eval.shape),
                 output_phase_shape=np.array(phase_eval.shape),
                 output_amplitude_crop=ampli_eval[0, 512:576, 512:576],
                 output_amplitude_mean=ampli_eval.mean())
        print("    [OK] solvers/evaluate.npz")

    else:
        print("    [SKIP] model_weights.pth not found - skipping network fixtures")

    # --- G_Renderer fixture ---
    torch.manual_seed(42)
    renderer = G_Renderer(in_dim=32, hidden_dim=32, num_layers=2, out_dim=1)
    renderer.eval()
    rng = np.random.default_rng(0)
    input_data = rng.standard_normal((4, 32)).astype(np.float32)
    with torch.no_grad():
        out = renderer(torch.from_numpy(input_data))
    torch.save(renderer.state_dict(),
               os.path.join(NETWORK_DIR, "g_renderer_state.pth"))
    np.savez(os.path.join(NETWORK_DIR, "g_renderer.npz"),
             input_data=input_data,
             output_shape=np.array(list(out.shape)),
             output_data=out.numpy())
    print("    [OK] network/g_renderer.npz + g_renderer_state.pth")


def main():
    print("Generating fixtures for fpm_inr_reconstruction ...")

    metadata, raw_data, optical_params, pupil_data, z_params = generate_preprocessing_fixtures()
    generate_visualization_fixtures()

    if optical_params is not None and pupil_data is not None:
        try:
            generate_cuda_fixtures(metadata, optical_params, pupil_data, z_params)
        except Exception as e:
            print(f"  [ERROR] CUDA fixtures failed: {e}")
    else:
        print("  [SKIP] CUDA fixtures (no raw data available)")

    print("Done! All fixtures saved to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
