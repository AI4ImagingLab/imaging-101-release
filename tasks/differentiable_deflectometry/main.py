"""
Differentiable Refractive Deflectometry - Main Pipeline
========================================================

Reproduces the experimental results from:
Wang et al., "Towards self-calibrated lens metrology by differentiable
refractive deflectometry", OSA Optics Express, 2021.

Pipeline: calibration -> fringe analysis -> LM optimization -> visualization
"""
import json
import os

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.image import imread

from src.physics_model import (
    init_device, Lensgroup, Scene, Screen, Transformation, set_texture,
    get_nested_attr,
)
from src.preprocessing import (
    load_calibration, crop_images, get_crop_offset, solve_for_intersections,
    compute_mount_origin, prepare_measurement_images, Fringe,
)
from src.solvers import LMSolver
from src.visualization import (
    plot_image_comparison, plot_loss_curve, plot_spot_diagram,
    compute_metrics, save_metrics,
)

# ---- Solver parameters (from Wang et al. 2021) ----
_LM_PARAMS = {
    'lambda_init': 0.1,
    'mu': 2.0,
    'regularization': 'diag',
    'max_iter': 100,
}
_INITIAL_PARAMS = {
    'c0': 0.001,
    'c1': 0.001,
    'd1': 3.0,
    'theta_x': 0.0,
    'theta_y': 0.0,
}
_DIFF_NAMES = [
    'lensgroup.surfaces[0].c',
    'lensgroup.surfaces[1].c',
    'lensgroup.surfaces[1].d',
    'lensgroup.origin',
    'lensgroup.theta_x',
    'lensgroup.theta_y',
]


def main():
    # Load config
    with open('data/meta_data.json') as f:
        config = json.load(f)

    device = init_device()

    # ---- Step 1: Load calibration ----
    print("Loading calibration data.")
    cameras, screen, p_rotation, lut_data = load_calibration(
        calibration_path='data/calibration',
        rotation_path='data/calibration/rotation.mat',
        lut_path='data/calibration/gammas.mat',
        scale=1.0,
        device=device,
    )

    # ---- Step 2: Build scene with lensgroup ----
    print("Initialize a DiffMetrology scene.")
    origin_shift = np.array(config['calibration']['origin_shift'])
    scene = Scene(cameras, screen, device=device)

    # Compute mount geometry
    origin_mount = compute_mount_origin(cameras, p_rotation, device=device)
    scene.lensgroup = Lensgroup(origin_mount, origin_shift, 0.0, 0.0, 0.0, device)

    # Load lens file
    lens_file = config['lens']['file']
    scene.lensgroup.load_file(lens_file, lens_dir='data/lenses/')

    print("Ground Truth Lens Parameters:")
    for i in range(len(scene.lensgroup.surfaces)):
        print(f"  Surface[{i}] radius of curvature: {1.0 / scene.lensgroup.surfaces[i].c.item()}")
    print(f"  Thickness: {scene.lensgroup.surfaces[1].d}")

    # ---- Step 3: Crop and set filmsize ----
    filmsize = np.array(config['imaging']['filmsize'])
    crop_offset = get_crop_offset(filmsize)
    for cam in scene.cameras:
        cam.filmsize = filmsize
        cam.crop_offset = torch.Tensor(crop_offset).to(device)

    # ---- Step 4: Load measurement data ----
    print("Loading measurement data.")
    data = np.load('data/raw_data.npz')
    imgs = crop_images(data['imgs'][0].astype(np.float64), filmsize)
    refs = crop_images(data['refs'][0].astype(np.float64), filmsize)
    del data

    # ---- Step 5: Solve for intersection points ----
    print("Solving for intersection points from fringe images.")
    Ts = np.array(config['imaging']['sinusoid_periods'])
    t_idx = config['imaging']['sinusoid_period_index']
    ps_cap, valid_cap, C = solve_for_intersections(imgs, refs, Ts[t_idx:], scene, device)

    # ---- Step 6: Set display pattern ----
    xs = config['imaging']['sinusoid_xs']
    sinusoid_path = 'data/sinusoids'
    ims = [np.mean(imread(sinusoid_path + '/' + str(x) + '.png'), axis=-1) for x in xs]
    ims = np.array([im / im.max() for im in ims])
    ims = np.sum(ims, axis=0)
    set_texture(scene, ims, device)
    del ims

    # Apply calibrated texture shift
    texture_shift = config['calibration']['texture_shift']
    scene.screen.texture_shift = torch.Tensor(texture_shift).to(device)

    # ---- Step 7: Recompute mount origin from measurement ----
    print("Shift origin by estimated value.")
    origin = compute_mount_origin(cameras, C, device=device, verbose=True)
    scene.lensgroup.origin = torch.Tensor(origin).to(device)
    scene.lensgroup.update()

    # ---- Step 8: Prepare measurement images ----
    print("Preparing measurement images for visualization.")
    FR = Fringe()
    a_cap, b_cap, psi_cap = FR.solve(imgs)
    I0 = prepare_measurement_images(imgs, xs, valid_cap, a_cap, device)

    # ---- Step 9: Define forward model and utilities ----
    angle = 0.0

    def forward():
        scene.lensgroup.update(_y=angle)
        ps = torch.stack(scene.trace(with_element=True, mask=valid_cap)[0])[..., 0:2]
        scene.lensgroup.update(_y=-angle)
        return ps

    def render():
        scene.lensgroup.update(_y=angle)
        I = valid_cap * torch.stack(scene.render(with_element=True))
        scene.lensgroup.update(_y=-angle)
        I[torch.isnan(I)] = 0.0
        return I

    def loss_fn(ps):
        return torch.sum((ps[valid_cap, ...] - ps_cap[valid_cap, ...])**2, axis=-1).mean()

    def residual_fn(ps):
        b = valid_cap[..., None] * (ps_cap - ps)
        b[torch.isnan(b)] = 0.0
        return b

    # ---- Step 10: Initialize lens parameters ----
    print("Initialize lens parameters.")
    scene.lensgroup.surfaces[0].c = torch.Tensor([_INITIAL_PARAMS['c0']]).to(device)
    scene.lensgroup.surfaces[1].c = torch.Tensor([_INITIAL_PARAMS['c1']]).to(device)
    scene.lensgroup.surfaces[1].d = torch.Tensor([_INITIAL_PARAMS['d1']]).to(device)
    scene.lensgroup.theta_x = torch.Tensor([_INITIAL_PARAMS['theta_x']]).to(device)
    scene.lensgroup.theta_y = torch.Tensor([_INITIAL_PARAMS['theta_y']]).to(device)
    scene.lensgroup.update()

    # Visualize initial state
    print("Visualize initial status.")
    ps_initial = forward()
    I_initial = render()
    os.makedirs('evaluation/reference_outputs', exist_ok=True)
    plot_image_comparison(I0, I_initial, valid_cap, filename_prefix='evaluation/reference_outputs/initial')
    plot_spot_diagram(ps_cap, ps_initial, valid_cap, scene.camera_count,
                      filename='evaluation/reference_outputs/spot_initial.jpg')
    plt.close('all')

    # ---- Step 11: Run LM optimization ----
    print("Running Levenberg-Marquardt optimization.")
    solver = LMSolver(
        lamb=_LM_PARAMS['lambda_init'],
        mu=_LM_PARAMS['mu'],
        regularization=_LM_PARAMS['regularization'],
        max_iter=_LM_PARAMS['max_iter'],
    )
    loss_history = solver.optimize(
        forward_fn=forward,
        scene=scene,
        param_names=_DIFF_NAMES,
        residual_fn=residual_fn,
        device=device,
    )

    print("Done. Show results:")
    for i in range(len(scene.lensgroup.surfaces)):
        print(f"  Surface[{i}] radius of curvature: {1.0 / scene.lensgroup.surfaces[i].c.item()}")
    print(f"  Thickness: {scene.lensgroup.surfaces[1].d}")

    # ---- Step 12: Visualize optimized state ----
    print("Visualize optimized status.")
    ps_optimized = forward()
    I_optimized = render()
    plot_image_comparison(I0, I_optimized, valid_cap, filename_prefix='evaluation/reference_outputs/optimized')
    plot_spot_diagram(ps_cap, ps_optimized, valid_cap, scene.camera_count,
                      filename='evaluation/reference_outputs/spot_optimized.jpg')
    plot_loss_curve(loss_history, filename='evaluation/reference_outputs/loss_curve.jpg')
    plt.close('all')

    # ---- Step 13: Compute and save metrics ----
    print("Computing metrics.")
    gt_data = np.load('data/ground_truth.npz')
    gt = {k: float(gt_data[k][0]) for k in gt_data}
    recovered = {
        'surface_0_roc_mm': 1.0 / scene.lensgroup.surfaces[0].c.item(),
        'surface_1_roc_mm': 1.0 / scene.lensgroup.surfaces[1].c.item(),
        'thickness_mm': scene.lensgroup.surfaces[1].d.item(),
    }

    metrics = compute_metrics(
        ps_modeled=ps_optimized,
        ps_measured=ps_cap,
        valid=valid_cap,
        loss_history=loss_history,
        recovered_params=recovered,
        gt_params=gt,
    )

    # Add tilt angles
    metrics['theta_x_deg'] = scene.lensgroup.theta_x.item()
    metrics['theta_y_deg'] = scene.lensgroup.theta_y.item()

    save_metrics(metrics, 'evaluation/reference_outputs/metrics.json')
    np.save('evaluation/reference_outputs/loss_history.npy', np.array(loss_history))

    # Save optimized params
    optimized_params = {
        'surface_0_c': scene.lensgroup.surfaces[0].c.item(),
        'surface_1_c': scene.lensgroup.surfaces[1].c.item(),
        'surface_1_d': scene.lensgroup.surfaces[1].d.item(),
        'origin': scene.lensgroup.origin.cpu().detach().numpy().tolist(),
        'theta_x': scene.lensgroup.theta_x.item(),
        'theta_y': scene.lensgroup.theta_y.item(),
    }
    with open('evaluation/reference_outputs/optimized_params.json', 'w') as f:
        json.dump(optimized_params, f, indent=2)

    print("\nFinal Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    T = ps_optimized - ps_cap
    E = torch.sqrt(torch.sum(T[valid_cap, ...]**2, axis=-1)).mean()
    print("\nMean displacement error = {:.2f} um".format(E.item() * 1e3))


if __name__ == '__main__':
    main()
