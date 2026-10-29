import numpy as np,torch
from pathlib import Path
from src.preprocessing import load_data,subtract_dark
from src.physics_model import apply_matrix,inverse_mueller
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_original_stages_and_pipeline():
    torch.set_num_threads(1);data=load_data(ROOT/'data/raw_data.npz');f=np.load(ROOT/'test/fixtures/upstream_stages.npz');raw,bg=subtract_dark(data)
    cal=torch.linalg.pinv(torch.tensor(data['instrument_matrix'][0],dtype=torch.float32));s=apply_matrix(cal,torch.from_numpy(raw));sb=apply_matrix(cal,torch.from_numpy(bg));mi=inverse_mueller(sb)
    np.testing.assert_allclose(cal.numpy(),f['output_analysis_matrix'],rtol=5e-5,atol=2e-5)
    np.testing.assert_allclose(s.numpy()[:,:,:,:16,:16],f['output_stokes'],rtol=5e-5,atol=2e-5)
    np.testing.assert_allclose(sb.numpy()[:,:16,:16],f['output_background_stokes'],rtol=5e-5,atol=2e-5)
    np.testing.assert_allclose(mi.numpy()[:,:,:16,:16],f['output_inverse_mueller'],rtol=5e-5,atol=2e-5)
    ref=np.load(ROOT/'evaluation/baseline_reference.npz');out=reconstruct(data)
    for k in out:np.testing.assert_allclose(out[k],ref[k],rtol=5e-5,atol=2e-5)
