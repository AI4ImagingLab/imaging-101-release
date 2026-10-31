"""Generate all fixture files for plane_wave_ultrasound tests.

Reads the src code and test expectations to produce:
  - param_erm_velocity.npz, output_erm_velocity.npy
  - param_stolt_fkz.npz, output_stolt_fkz.npy
  - param_steering_delay.npz, output_steering_delay.npy
  - input_remove_dc.npy, output_remove_dc.npy
  - input_toy_raw_data.npz, input_toy_meta_data.json, output_toy_rf_fibers.npy
  - input_fkmig_case.npz, output_fkmig_case.npz
  - input_compound_case.npz, output_compound_case.npz
  - input_envelope_bmode.npy, output_envelope_bmode.npy
  - input_psf_case.npz, output_psf_fwhm.npy
  - input_cnr_case.npz, output_cnr.npy
"""
import sys
import json
import pathlib
import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.physics_model import erm_velocity, stolt_fkz, steering_delay
from src.preprocessing import remove_dc, load_dataset
from src.solvers import fkmig, coherent_compound
from src.visualization import envelope_bmode, measure_psf_fwhm, measure_cnr

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def gen_physics_model_fixtures():
    """Generate fixtures for test_physics_model.py."""
    rng = np.random.default_rng(42)

    # --- erm_velocity ---
    c = 1540.0
    txangle = 0.1
    result = erm_velocity(c, txangle)
    np.savez(FIXTURES_DIR / "param_erm_velocity.npz", c=c, txangle=txangle)
    np.save(FIXTURES_DIR / "output_erm_velocity.npy", np.array(result))
    print("  erm_velocity: OK")

    # --- stolt_fkz ---
    nf, nkx = 16, 12
    f = rng.uniform(1e6, 10e6, (nf, nkx))
    Kx = rng.uniform(-1000, 1000, (nf, nkx))
    c_fkz = 1540.0
    txangle_fkz = 0.05
    result_fkz = stolt_fkz(f, Kx, c_fkz, txangle_fkz)
    np.savez(FIXTURES_DIR / "param_stolt_fkz.npz",
             f=f, Kx=Kx, c=c_fkz, txangle=txangle_fkz)
    np.save(FIXTURES_DIR / "output_stolt_fkz.npy", result_fkz)
    print("  stolt_fkz: OK")

    # --- steering_delay ---
    nx = 64
    pitch = 2.98e-4
    c_sd = 1540.0
    txangle_sd = -0.1
    t0 = 0.0
    result_sd = steering_delay(nx, pitch, c_sd, txangle_sd, t0)
    np.savez(FIXTURES_DIR / "param_steering_delay.npz",
             nx=nx, pitch=pitch, c=c_sd, txangle=txangle_sd, t0=t0)
    np.save(FIXTURES_DIR / "output_steering_delay.npy", result_sd)
    print("  steering_delay: OK")


def gen_preprocessing_fixtures():
    """Generate fixtures for test_preprocessing.py."""
    rng = np.random.default_rng(123)

    # --- remove_dc ---
    rf_input = rng.standard_normal((64, 32)) * 1000
    rf_output = remove_dc(rf_input)
    np.save(FIXTURES_DIR / "input_remove_dc.npy", rf_input)
    np.save(FIXTURES_DIR / "output_remove_dc.npy", rf_output)
    print("  remove_dc: OK")

    # --- load_dataset (toy data) ---
    # Create a toy raw_data.npz and meta_data.json that load_dataset can read
    nt, nx_arr, n_angles = 32, 16, 3
    RF_fibers = rng.standard_normal((1, nt, nx_arr, n_angles)).astype(np.float64)
    RF_cysts = rng.standard_normal((1, nt, nx_arr, n_angles)).astype(np.float64)

    np.savez(FIXTURES_DIR / "input_toy_raw_data.npz",
             RF_fibers=RF_fibers, RF_cysts=RF_cysts)

    meta = {
        "dataset_1": {
            "c": 1540.0,
            "fs": 20000000.0,
            "pitch": 2.98e-4,
            "TXangle_rad": [-0.1, 0.0, 0.1],
            "t0": 0.0,
        },
        "dataset_2": {
            "c": 1540.0,
            "fs": 20000000.0,
            "pitch": 2.98e-4,
            "TXangle_rad": [-0.1, 0.0, 0.1],
            "t0": 0.0,
        },
    }
    with open(FIXTURES_DIR / "input_toy_meta_data.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Load using the actual function to get expected output
    rf, params = load_dataset(
        str(FIXTURES_DIR / "input_toy_raw_data.npz"),
        str(FIXTURES_DIR / "input_toy_meta_data.json"),
        dataset="fibers",
    )
    np.save(FIXTURES_DIR / "output_toy_rf_fibers.npy", rf)
    print("  load_dataset: OK")


def gen_solvers_fixtures():
    """Generate fixtures for test_solvers.py."""
    rng = np.random.default_rng(456)

    # --- fkmig ---
    # Use small signals to keep it fast
    nt, nx = 64, 16
    SIG = rng.standard_normal((nt, nx)).astype(np.float64)
    fs = 20e6
    pitch = 2.98e-4
    txangle = 0.05
    c = 1540.0
    t0 = 0.0

    x, z, mig = fkmig(SIG, fs, pitch, TXangle=txangle, c=c, t0=t0)
    np.savez(FIXTURES_DIR / "input_fkmig_case.npz",
             SIG=SIG, fs=fs, pitch=pitch, txangle=txangle, c=c, t0=t0)
    np.savez(FIXTURES_DIR / "output_fkmig_case.npz", x=x, z=z, mig=mig)
    print("  fkmig: OK")

    # --- coherent_compound ---
    n_angles = 3
    RF = rng.standard_normal((nt, nx, n_angles)).astype(np.float64)
    TXangles = np.array([-0.1, 0.0, 0.1])

    x_c, z_c, compound = coherent_compound(RF, fs, pitch, TXangles, c=c, t0=t0)
    np.savez(FIXTURES_DIR / "input_compound_case.npz",
             RF=RF, fs=fs, pitch=pitch, TXangles=TXangles, c=c, t0=t0)
    np.savez(FIXTURES_DIR / "output_compound_case.npz",
             x=x_c, z=z_c, compound=compound)
    print("  coherent_compound: OK")


def gen_visualization_fixtures():
    """Generate fixtures for test_visualization.py."""
    rng = np.random.default_rng(789)

    # --- envelope_bmode ---
    nt, nx = 64, 16
    mig = rng.standard_normal((nt, nx)) + 1j * rng.standard_normal((nt, nx))
    bmode = envelope_bmode(mig, gamma=0.5)
    np.save(FIXTURES_DIR / "input_envelope_bmode.npy", mig)
    np.save(FIXTURES_DIR / "output_envelope_bmode.npy", bmode)
    print("  envelope_bmode: OK")

    # --- measure_psf_fwhm ---
    # Create a B-mode image with known point targets
    nt_psf, nx_psf = 128, 64
    x_arr = (np.arange(nx_psf) - (nx_psf - 1) / 2.0) * 2.98e-4
    z_arr = np.arange(nt_psf) * 1540.0 / 2.0 / 20e6

    # Bright points at known depths
    bmode_psf = rng.uniform(0.01, 0.1, (nt_psf, nx_psf))
    z_targets = [z_arr[30], z_arr[60], z_arr[90]]
    for zt in z_targets:
        iz = int(np.argmin(np.abs(z_arr - zt)))
        ix = nx_psf // 2
        # Create a Gaussian peak
        for dx in range(-5, 6):
            if 0 <= ix + dx < nx_psf:
                bmode_psf[iz, ix + dx] = 1.0 * np.exp(-0.5 * (dx / 1.5) ** 2)

    fwhms = measure_psf_fwhm(bmode_psf, x_arr, z_arr, z_targets)
    np.savez(FIXTURES_DIR / "input_psf_case.npz",
             bmode=bmode_psf, x=x_arr, z=z_arr, z_targets=np.array(z_targets))
    np.save(FIXTURES_DIR / "output_psf_fwhm.npy", np.array(fwhms))
    print("  measure_psf_fwhm: OK")

    # --- measure_cnr ---
    nt_cnr, nx_cnr = 128, 64
    x_cnr = (np.arange(nx_cnr) - (nx_cnr - 1) / 2.0) * 2.98e-4
    z_cnr = np.arange(nt_cnr) * 1540.0 / 2.0 / 20e6

    bmode_cnr = rng.uniform(0.5, 1.5, (nt_cnr, nx_cnr))

    # Create cyst-like regions (darker circles)
    cyst_centers = [(x_cnr[20], z_cnr[40]), (x_cnr[45], z_cnr[80])]
    X, Z = np.meshgrid(x_cnr, z_cnr)
    for (xc, zc) in cyst_centers:
        dist = np.sqrt((X - xc) ** 2 + (Z - zc) ** 2)
        bmode_cnr[dist < 2e-3] *= 0.3

    cyst_radius = 2e-3
    shell_inner = 2.5e-3
    shell_outer = 4e-3
    cnrs = measure_cnr(bmode_cnr, x_cnr, z_cnr, cyst_centers,
                       cyst_radius=cyst_radius,
                       shell_inner=shell_inner,
                       shell_outer=shell_outer)
    np.savez(FIXTURES_DIR / "input_cnr_case.npz",
             bmode=bmode_cnr, x=x_cnr, z=z_cnr,
             cyst_centers=np.array(cyst_centers),
             cyst_radius=cyst_radius,
             shell_inner=shell_inner,
             shell_outer=shell_outer)
    np.save(FIXTURES_DIR / "output_cnr.npy", np.array(cnrs))
    print("  measure_cnr: OK")


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating physics model fixtures...")
    gen_physics_model_fixtures()

    print("Generating preprocessing fixtures...")
    gen_preprocessing_fixtures()

    print("Generating solver fixtures...")
    gen_solvers_fixtures()

    print("Generating visualization fixtures...")
    gen_visualization_fixtures()

    print(f"\nAll fixtures saved to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
