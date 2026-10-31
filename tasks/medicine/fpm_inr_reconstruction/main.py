"""
FPM-INR: Fourier Ptychographic Microscopy with Implicit Neural Representations
===============================================================================

Pipeline entry point for reconstructing 3D image stacks from FPM measurements
using implicit neural representations.

Steps:
    1. Load and preprocess FPM measurement data
    2. Build the FPM forward model
    3. Train INR to reconstruct the 3D complex field
    4. Evaluate per-slice against ground truth
    5. Compute all-in-focus image and compare
    6. Visualize and save results

Usage:
    cd tasks/fpm_inr
    python main.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

from src.preprocessing import prepare_data, load_metadata, load_ground_truth
from src.physics_model import FPMForwardModel
from src.solvers import FullModel, save_model_with_required_grad, load_model_with_required_grad, FPMINRSolver
from src.visualization import (
    all_in_focus_normal_variance,
    plot_amplitude_phase,
    plot_per_slice_metrics,
    plot_gt_comparison,
    plot_allfocus_comparison,
    compute_metrics,
    compute_ssim_per_slice,
    compute_allfocus_l2,
)

# Reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_dir = "data"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load and preprocess data ─────────────────────────────────
    print("Step 1: Loading and preprocessing data...")
    data = prepare_data(data_dir=data_dir, device=device)
    metadata = data["metadata"]
    optical_params = data["optical_params"]
    pupil_data = data["pupil_data"]
    z_params = data["z_params"]
    Isum = data["Isum"]

    training_cfg = metadata["training"]
    M = optical_params["M"]
    N = optical_params["N"]
    MM = optical_params["MM"]
    NN = optical_params["NN"]
    MAGimg = optical_params["MAGimg"]
    ID_len = optical_params["ID_len"]

    print(f"  Image size: {M}x{N}, {ID_len} LEDs, upsampled to {MM}x{NN}")
    print(f"  Z range: [{z_params['z_min']}, {z_params['z_max']}] um, {z_params['num_z']} training planes")
    print(f"  Device: {device}")

    # ── Step 2: Build forward model ──────────────────────────────────────
    print("\nStep 2: Building forward model...")
    forward_model = FPMForwardModel(
        Pupil0=pupil_data["Pupil0"],
        kzz=pupil_data["kzz"],
        ledpos_true=optical_params["ledpos_true"],
        M=M,
        N=N,
        MAGimg=MAGimg,
    )

    # ── Step 3: Train INR ────────────────────────────────────────────────
    print(f"\nStep 3: Training FPM-INR ({training_cfg['num_epochs']} epochs)...")
    model = FullModel(
        w=MM,
        h=MM,
        num_feats=training_cfg["num_feats"],
        x_mode=metadata["num_modes"],
        y_mode=metadata["num_modes"],
        z_min=z_params["z_min"],
        z_max=z_params["z_max"],
        ds_factor=1,
        use_layernorm=training_cfg["use_layernorm"],
    ).to(device)

    solver = FPMINRSolver(
        num_epochs=training_cfg["num_epochs"],
        lr=training_cfg["lr"],
        lr_decay_step=training_cfg["lr_decay_step"],
        lr_decay_gamma=training_cfg["lr_decay_gamma"],
        use_amp=training_cfg["use_amp"],
        use_compile=training_cfg["use_compile"],
    )

    def vis_callback(epoch, amplitude, phase):
        plot_amplitude_phase(
            amplitude, phase, epoch=epoch,
            save_path=os.path.join(output_dir, f"e_{epoch}.png"),
        )

    train_results = solver.train(
        model=model,
        forward_model=forward_model,
        Isum=Isum,
        z_params=z_params,
        device=device,
        vis_callback=vis_callback,
    )
    print(f"  Final loss: {train_results['final_loss']:.4e}, PSNR: {train_results['final_psnr']:.2f} dB")

    # Save trained model
    model_path = os.path.join(output_dir, "model_weights.pth")
    save_model_with_required_grad(model, model_path)
    print(f"  Model saved to {model_path}")

    # ── Step 4: Evaluate per-slice against GT ────────────────────────────
    print("\nStep 4: Evaluating per-slice against ground truth...")
    gt_data = load_ground_truth(data_dir)
    gt_stack = gt_data["I_stack"]  # (H, W, n_z)
    gt_zvec = gt_data["zvec"]  # (n_z,)

    # Inference at GT z-positions
    pred_ampli, pred_phase = solver.evaluate(model, gt_zvec, device=device)
    # pred_ampli shape: (n_z, H, W) at upsampled resolution

    # Resize prediction to match GT resolution
    gt_hw = gt_stack.shape[0]
    pred_hw = pred_ampli.shape[1]
    if pred_hw != gt_hw:
        print(f"  Resizing prediction from {pred_hw} to {gt_hw}")
        pred_tensor = torch.from_numpy(pred_ampli).unsqueeze(1)
        pred_tensor = F.interpolate(
            pred_tensor, size=(gt_hw, gt_hw), mode="bilinear", align_corners=False
        )
        pred_ampli_resized = pred_tensor.squeeze(1).numpy()
    else:
        pred_ampli_resized = pred_ampli

    # GT is (H, W, Z) -> (Z, H, W)
    gt_ampli = np.transpose(gt_stack, (2, 0, 1))

    metrics = compute_metrics(pred_ampli_resized, gt_ampli)
    ssim_per_slice = compute_ssim_per_slice(metrics["pred_norm"], metrics["gt_norm"])

    print(f"  Overall PSNR: {metrics['psnr_overall']:.2f} dB")
    print(f"  Per-slice PSNR: {metrics['psnr_per_slice'].mean():.2f} +/- {metrics['psnr_per_slice'].std():.2f} dB")
    print(f"  Per-slice SSIM: {ssim_per_slice.mean():.4f} +/- {ssim_per_slice.std():.4f}")

    # Save per-slice visualizations
    plot_per_slice_metrics(
        gt_zvec, metrics["l2_per_slice"], metrics["psnr_per_slice"],
        ssim=ssim_per_slice,
        save_path=os.path.join(output_dir, "gt_comparison_metrics.png"),
    )
    plot_gt_comparison(
        metrics["pred_norm"], metrics["gt_norm"], gt_zvec,
        metrics["psnr_per_slice"], metrics["l2_per_slice"],
        save_path=os.path.join(output_dir, "gt_comparison_visual.png"),
    )

    # ── Step 5: Compute all-in-focus and compare ─────────────────────────
    print("\nStep 5: Computing all-in-focus images...")

    # Generate 161 z-slices from model at full resolution
    z_stack_slices = metadata["z_stack_slices"]
    z_eval = np.linspace(z_params["z_min"], z_params["z_max"], z_stack_slices)
    pred_ampli_full, _ = solver.evaluate(model, z_eval, device=device)

    # Normalize to [0, 1]
    pred_imgs = pred_ampli_full.squeeze()
    pred_imgs = (pred_imgs - pred_imgs.min()) / (pred_imgs.max() - pred_imgs.min())

    # Compute FPM-INR all-in-focus
    imgs_AIF_INR = np.moveaxis(pred_imgs, 0, -1)  # (H, W, n_z)
    aif_pred = all_in_focus_normal_variance(imgs_AIF_INR)

    # Compute GT all-in-focus
    gt_imgs = np.abs(gt_stack).astype("float32")
    gt_imgs = (gt_imgs - gt_imgs.min()) / (gt_imgs.max() - gt_imgs.min())
    aif_gt_lowres = all_in_focus_normal_variance(gt_imgs)
    aif_gt = zoom(aif_gt_lowres, zoom=MAGimg, order=1)

    # Compute L2 error
    aif_metrics = compute_allfocus_l2(aif_pred, aif_gt)
    print(f"  All-in-focus L2 error (MSE): {aif_metrics['mse']:.6e}")
    print(f"  All-in-focus PSNR: {aif_metrics['psnr']:.2f} dB")
    print(f"  Paper reports: L2 Error = 1.41e-3")

    plot_allfocus_comparison(
        aif_pred, aif_gt, aif_metrics["mse"],
        save_path=os.path.join(output_dir, "allfocus_comparison.png"),
    )

    # ── Step 6: Save results ─────────────────────────────────────────────
    print("\nStep 6: Saving results...")

    results = {
        "aif_l2_error": aif_metrics["mse"],
        "aif_psnr": aif_metrics["psnr"],
        "per_slice_psnr_mean": float(metrics["psnr_per_slice"].mean()),
        "per_slice_psnr_std": float(metrics["psnr_per_slice"].std()),
        "per_slice_ssim_mean": float(ssim_per_slice.mean()),
        "per_slice_ssim_std": float(ssim_per_slice.std()),
        "overall_mse": float(metrics["mse_overall"]),
        "overall_psnr": float(metrics["psnr_overall"]),
        "training_epochs": training_cfg["num_epochs"],
        "final_training_loss": train_results["final_loss"],
        "final_training_psnr": train_results["final_psnr"],
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Metrics saved to {metrics_path}")
    print(f"  Visualizations saved to {output_dir}/")
    print("\nDone!")

    return results


if __name__ == "__main__":
    main()
