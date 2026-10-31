#!/usr/bin/env python3
"""
Pipeline entry point for shapelet source reconstruction task.

Runs the full pipeline: preprocessing -> simulation -> reconstruction -> evaluation.
Saves all outputs and figures to evaluation/reference_outputs/.
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import numpy as np

# Add task root to path so src/ can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generate_data import generate_all_data, DEFAULT_PARAMS
from src.preprocessing import load_and_prepare_galaxy, reconstruct_from_shapelets
from src.visualization import (
    plot_shapelet_decomposition, plot_lensing_stages,
    plot_reconstruction, plot_unlensed_stages,
)


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(task_dir))
    image_path = os.path.join(repo_root, 'Data', 'Galaxies', 'ngc1300.jpg')
    output_dir = os.path.join(task_dir, 'evaluation', 'reference_outputs')

    print("=" * 70)
    print("Shapelet Source Reconstruction Pipeline")
    print("=" * 70)

    # Generate all data
    metrics = generate_all_data(image_path, output_dir, seed=42)

    # Load outputs for plotting
    raw = np.load(os.path.join(output_dir, 'raw_data.npz'))
    lens = np.load(os.path.join(output_dir, 'lensing_outputs.npz'))
    dc = np.load(os.path.join(output_dir, 'deconv_outputs.npz'))

    # Figure 1: Shapelet decomposition
    print("\nGenerating figures...")
    plot_shapelet_decomposition(
        raw['ngc_square'], raw['ngc_conv'], raw['ngc_resized'], raw['image_reconstructed'],
        save_path=os.path.join(output_dir, 'fig1_shapelet_decomposition.png'))

    # Figure 2: Lensing stages
    plot_lensing_stages(
        [lens['image_hr_nolens'], lens['image_hr_lensed'],
         lens['image_hr_conv'], lens['image_no_noise'], lens['image_real']],
        ["intrinsic source", "+ lensing effect", "+ convolution", "+ pixelisation", "+ noise"],
        save_path=os.path.join(output_dir, 'fig2_lensing_simulation.png'))

    # Figure 3: Source reconstruction
    plot_reconstruction([
        (lens['image_real'], "Input image"),
        (lens['model_lens'], "Reconstructed image"),
        (lens['residuals_lens'], "Image residuals"),
        (lens['image_hr_nolens'], "Input source"),
        (lens['source_recon_2d'], "Reconstructed source"),
        (lens['source_recon_2d'] - lens['image_hr_nolens'], "Source residuals"),
    ], save_path=os.path.join(output_dir, 'fig3_source_reconstruction.png'))

    # Figure 4: Deconvolution setup
    plot_unlensed_stages(
        [dc['image_hr_dc'], dc['image_hr_conv_dc'], dc['image_no_noise_dc'], dc['image_real_dc']],
        save_path=os.path.join(output_dir, 'fig4_deconvolution_setup.png'))

    # Figure 5: Deconvolution reconstruction
    plot_reconstruction([
        (dc['image_real_dc'], "Input image"),
        (dc['model_dc'], "Reconstructed image"),
        (dc['residuals_dc'], "Image residuals"),
        (dc['image_hr_dc'], "Input unconvolved"),
        (dc['source_deconv_2d'], "de-convolved image"),
        (dc['source_deconv_2d'] - dc['image_hr_dc'], "de-convolution residuals"),
    ], save_path=os.path.join(output_dir, 'fig5_deconvolution_result.png'))

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {output_dir}")
    print(f"Metrics: {metrics}")
    print("=" * 70)


if __name__ == '__main__':
    main()
