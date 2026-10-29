#!/usr/bin/env python
"""Generate test fixtures for differentiable_deflectometry.

Creates fixtures in evaluation/fixtures/ that the tests load.
"""

import os
import sys
import numpy as np
import torch

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

FIXTURES_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

from src.physics_model import (
    Ray, Transformation, Material, Aspheric, normalize,
    rodrigues_rotation_matrix, get_nested_attr, set_nested_attr,
)
from src.preprocessing import Fringe, crop_images, get_crop_offset
from src.solvers import LMSolver, change_parameters


def main():
    print("Generating fixtures for differentiable_deflectometry ...")

    # ============ physics_model fixtures ============

    # --- Ray evaluation ---
    o = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    d = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    wavelength = 500.0
    t_val = 10.0
    ray = Ray(torch.Tensor(o), torch.Tensor(d), wavelength)
    p = ray(torch.Tensor([t_val]))
    expected_point = p.squeeze().detach().numpy()
    np.savez(os.path.join(FIXTURES_DIR, "input_ray_evaluation.npz"),
             o=o, d=d, wavelength=wavelength, t=t_val)
    np.savez(os.path.join(FIXTURES_DIR, "output_ray_evaluation.npz"),
             expected_point=expected_point)
    print("  [OK] input/output_ray_evaluation.npz")

    # --- Transformation ---
    identity_R = np.eye(3, dtype=np.float32)
    identity_t = np.zeros(3, dtype=np.float32)
    test_point = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    T_id = Transformation(torch.Tensor(identity_R), torch.Tensor(identity_t))
    identity_expected = T_id.transform_point(torch.Tensor(test_point)).detach().numpy()

    rotation_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rotation_angle = float(np.pi / 4)
    translation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inverse_test_point = np.array([5.0, -3.0, 2.0], dtype=np.float32)

    np.savez(os.path.join(FIXTURES_DIR, "input_transformation.npz"),
             identity_R=identity_R,
             identity_t=identity_t,
             test_point=test_point,
             rotation_axis=rotation_axis,
             rotation_angle=rotation_angle,
             translation=translation,
             inverse_test_point=inverse_test_point)
    np.savez(os.path.join(FIXTURES_DIR, "output_transformation.npz"),
             identity_expected=identity_expected)
    print("  [OK] input/output_transformation.npz")

    # --- Material ---
    nbk7_wavelength = 589.3
    air_wavelength = 550.0

    mat_nbk7 = Material('N-BK7')
    nbk7_ior = mat_nbk7.ior(torch.Tensor([nbk7_wavelength]))
    mat_air = Material('air')
    air_ior = mat_air.ior(torch.Tensor([air_wavelength]))

    np.savez(os.path.join(FIXTURES_DIR, "input_material.npz"),
             nbk7_wavelength=nbk7_wavelength,
             air_wavelength=air_wavelength)
    np.savez(os.path.join(FIXTURES_DIR, "output_material.npz"),
             nbk7_ior_approx=float(nbk7_ior) if hasattr(nbk7_ior, 'item') else float(nbk7_ior),
             nbk7_tolerance=0.01,
             air_ior_approx=float(air_ior) if hasattr(air_ior, 'item') else float(air_ior),
             air_tolerance=0.01)
    print("  [OK] input/output_material.npz")

    # --- Aspheric flat surface ---
    np.savez(os.path.join(FIXTURES_DIR, "param_aspheric_flat.npz"),
             r=0.0, d=0.0, c=0.0, k=0.0)
    flat_x = np.array([0.0, 1.0, 5.0, -3.0], dtype=np.float32)
    flat_y = np.array([0.0, 0.0, 2.0, -1.0], dtype=np.float32)
    surface_flat = Aspheric(r=0.0, d=0.0, c=0.0, k=0.0)
    flat_z = surface_flat.surface(torch.Tensor(flat_x), torch.Tensor(flat_y)).detach().numpy()
    np.savez(os.path.join(FIXTURES_DIR, "input_aspheric_flat.npz"),
             x=flat_x, y=flat_y)
    np.savez(os.path.join(FIXTURES_DIR, "output_aspheric_flat.npz"),
             expected_z=flat_z)
    print("  [OK] param/input/output_aspheric_flat.npz")

    # --- Aspheric spherical surface ---
    c_val = 1.0 / 50.0  # R=50mm
    np.savez(os.path.join(FIXTURES_DIR, "param_aspheric_spherical.npz"),
             r=25.0, d=0.0, c=c_val, k=0.0)
    surface_sph = Aspheric(r=25.0, d=0.0, c=c_val, k=0.0)
    z_at_5 = surface_sph.surface(torch.Tensor([5.0]), torch.Tensor([0.0])).item()
    np.savez(os.path.join(FIXTURES_DIR, "input_aspheric_spherical.npz"),
             x_origin=0.0, y_origin=0.0,
             x_test=5.0, y_test=0.0)
    np.savez(os.path.join(FIXTURES_DIR, "output_aspheric_spherical.npz"),
             z_at_x5=z_at_5)
    print("  [OK] param/input/output_aspheric_spherical.npz")

    # --- Aspheric ray intersection ---
    c_int = 1.0 / 100.0
    np.savez(os.path.join(FIXTURES_DIR, "param_aspheric_intersection.npz"),
             r=50.0, d=0.0, c=c_int, k=0.0)
    o_int = np.array([[0.0, 0.0, -100.0]], dtype=np.float32)
    d_int = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    surface_int = Aspheric(r=50.0, d=0.0, c=c_int, k=0.0)
    valid, p_int = surface_int.newtons_method(1e5, torch.Tensor(o_int), torch.Tensor(d_int))
    z_approx = p_int[0, 2].item()
    np.savez(os.path.join(FIXTURES_DIR, "input_aspheric_intersection.npz"),
             o=o_int, d=d_int)
    np.savez(os.path.join(FIXTURES_DIR, "output_aspheric_intersection.npz"),
             expected_z_approx=z_approx,
             z_tolerance=0.1)
    print("  [OK] param/input/output_aspheric_intersection.npz")

    # ============ preprocessing fixtures ============

    # --- Fringe solve ---
    N_cam = 2
    H, W = 32, 32
    true_a = 100.0
    true_b = 50.0
    x_coords = np.arange(W, dtype=np.float64)
    y_coords = np.arange(H, dtype=np.float64)
    true_phase_x = np.tile(2.0 * np.pi * x_coords / W, (N_cam, H, 1))
    true_phase_y = np.tile(2.0 * np.pi * y_coords[:, None] / H, (N_cam, 1, W))

    phase_shifts = [0, np.pi/2, np.pi, 3*np.pi/2]
    imgs = np.zeros((8, N_cam, H, W), dtype=np.float64)
    for i, ps in enumerate(phase_shifts):
        imgs[i] = true_a + true_b * np.cos(true_phase_x + ps)
    for i, ps in enumerate(phase_shifts):
        imgs[4 + i] = true_a + true_b * np.cos(true_phase_y + ps)

    # Run the actual Fringe.solve to get the exact output (arctan2 wrapping)
    FR = Fringe()
    a_out, b_out, psi_out = FR.solve(imgs)

    np.savez(os.path.join(FIXTURES_DIR, "input_fringe_solve.npz"),
             imgs=imgs)
    np.savez(os.path.join(FIXTURES_DIR, "param_fringe_solve.npz"),
             N_cam=N_cam, H=H, W=W)
    np.savez(os.path.join(FIXTURES_DIR, "output_fringe_solve.npz"),
             true_a=true_a,
             true_phase_x=psi_out[0],
             true_phase_y=psi_out[1])
    print("  [OK] fringe_solve fixtures")

    # --- Crop images ---
    filmsize = np.array([768, 768])
    rng = np.random.default_rng(0)
    imgs_crop = rng.random((2, 2, 2048, 2048))
    cropped = crop_images(imgs_crop, filmsize)
    center_pixel = np.array([384, 384])
    center_val = cropped[0, 0, center_pixel[0], center_pixel[1]]
    np.savez(os.path.join(FIXTURES_DIR, "input_crop_images.npz"),
             imgs=imgs_crop, filmsize=filmsize)
    np.savez(os.path.join(FIXTURES_DIR, "output_crop_images.npz"),
             center_pixel=center_pixel,
             expected_center_value=center_val)
    print("  [OK] crop_images fixtures")

    # --- Crop offset ---
    filmsize_co = np.array([768, 768])
    offset = get_crop_offset(filmsize_co)
    np.savez(os.path.join(FIXTURES_DIR, "config_crop_offset.npz"),
             filmsize=filmsize_co,
             expected_offset=offset)
    print("  [OK] config_crop_offset.npz")

    # ============ solvers fixtures ============

    # --- Jacobian linear ---
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x_jac = np.array([1.0, 1.0], dtype=np.float32)
    np.savez(os.path.join(FIXTURES_DIR, "input_jacobian_linear.npz"),
             A=A, x=x_jac)
    np.savez(os.path.join(FIXTURES_DIR, "output_jacobian_linear.npz"),
             expected_J=A)
    print("  [OK] jacobian_linear fixtures")

    # --- Jacobian quadratic ---
    x_quad = np.array([3.0], dtype=np.float32)
    np.savez(os.path.join(FIXTURES_DIR, "input_jacobian_quadratic.npz"),
             x=x_quad)
    np.savez(os.path.join(FIXTURES_DIR, "output_jacobian_quadratic.npz"),
             expected_J=6.0)
    print("  [OK] jacobian_quadratic fixtures")

    # --- LM quadratic ---
    x_init = np.array([0.0, 0.0], dtype=np.float32)
    target = np.array([3.0, -2.0], dtype=np.float32)
    np.savez(os.path.join(FIXTURES_DIR, "input_lm_quadratic.npz"),
             x_init=x_init, target=target,
             lamb=0.1, n_iters=100)
    np.savez(os.path.join(FIXTURES_DIR, "output_lm_quadratic.npz"),
             expected_x=target,
             tolerance=0.1)
    print("  [OK] lm_quadratic fixtures")

    # --- Change parameters ---
    x_cp_init = np.array([1.0], dtype=np.float32)
    y_cp_init = np.array([2.0], dtype=np.float32)
    delta_x = np.array([0.5], dtype=np.float32)
    delta_y = np.array([-0.3], dtype=np.float32)

    np.savez(os.path.join(FIXTURES_DIR, "input_change_params.npz"),
             x_init=x_cp_init, y_init=y_cp_init,
             delta_x=delta_x, delta_y=delta_y)

    # Test add (sign=True)
    class MockScene:
        def __init__(self):
            self.x = torch.Tensor(x_cp_init.copy())
            self.y = torch.Tensor(y_cp_init.copy())

    scene = MockScene()
    result = change_parameters(
        scene, ['x', 'y'],
        [torch.Tensor(delta_x), torch.Tensor(delta_y)],
        sign=True
    )
    np.savez(os.path.join(FIXTURES_DIR, "output_change_params_add.npz"),
             expected_x=result[0].item(),
             expected_y=result[1].item())

    # Test subtract (sign=False)
    scene2 = MockScene()
    change_parameters(scene2, ['x'], [torch.Tensor(delta_x)], sign=False)
    np.savez(os.path.join(FIXTURES_DIR, "output_change_params_subtract.npz"),
             expected_x=get_nested_attr(scene2, 'x').item())
    print("  [OK] change_params fixtures")

    print("Done! All fixtures saved to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
