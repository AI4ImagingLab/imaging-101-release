from pathlib import Path
import numpy as np
from src.physics_model import spacing,build_rhs,matrix_apply
from src.solvers import solve,sequential_dot
ROOT=Path(__file__).resolve().parents[2]

def test_frozen_original_stages():
    with np.load(ROOT/"test/fixtures/upstream_stages.npz") as z:
        h=spacing(z["input_coordinate"]);b=build_rhs(z["input_gradient"],h)
        assert np.array_equal(b,z["output_rhs"])
        assert np.array_equal(matrix_apply(z["input_probe"],h),z["output_matvec"])
        a=matrix_apply(b,h);assert np.array_equal(a,z["output_matvec_rhs"])
        rr=sequential_dot(b,b);alpha=np.float32(rr/sequential_dot(b,a))
        assert rr==z["output_residual_squared"] and alpha==z["output_alpha"]
        assert np.array_equal(alpha*b,z["output_first_update"])

def test_pipeline_and_grid_original_parity():
    for n in [129,257]:
        with np.load(ROOT/f"test/fixtures/grid_{n}.npz") as z:
            p,h=solve(z["input_gradient"],z["input_coordinate"])
            assert np.array_equal(p,z["output_pressure"])
            assert h[-1,1]<=1e-4 or h[-1,2]<=1e-6
            assert p[1,1]==0

def test_fresh_full_native_original_parity():
    # main's fresh native execution is archived in the emitted standard result;
    # the active notebook independently repeats the entire full path.
    with np.load(ROOT/"output/reconstruction.npz") as p,np.load(ROOT/"notebooks/reference_outputs/upstream_measurements.npz") as u:
        assert np.array_equal(p["reconstruction"],u["reconstruction"])
