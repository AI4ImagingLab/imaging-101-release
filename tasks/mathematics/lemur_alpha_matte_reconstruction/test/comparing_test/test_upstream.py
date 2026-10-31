from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from src.preprocessing import load_data,partition_trimap
from src.physics_model import matting_laplacian
from src.solvers import reconstruct
from src.main import run

ROOT=Path(__file__).resolve().parents[2]


def test_original_partition_laplacian_and_all_cg_stages():
    image,trimap=load_data(ROOT/'data/raw_data.npz')
    alpha,diagnostics=reconstruct(image,trimap,capture_steps=[1,2,3,10,100])
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        fg,bg,known,unknown=partition_trimap(trimap)
        for key,actual in [('fg',fg),('known',known),('unknown',unknown)]:np.testing.assert_array_equal(actual,z[key])
        source=csr_matrix((z['laplacian_data'],z['laplacian_indices'],z['laplacian_indptr']),shape=diagnostics['laplacian'].shape)
        difference=source-diagnostics['laplacian'];assert np.max(abs(difference.data),initial=0)<1e-11
        np.testing.assert_allclose(diagnostics['laplacian']@z['probe'].ravel(),z['laplacian_probe'],rtol=1e-8,atol=1e-11)
        np.testing.assert_allclose(diagnostics['rhs'],z['rhs'],rtol=1e-8,atol=1e-11)
        for k,x in diagnostics['captured'].items():np.testing.assert_allclose(x,z['cg_x_'+str(k)],rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(diagnostics['history'],z['cg_history'],rtol=1e-7,atol=1e-8)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:reference=z['reconstruction'][0]
    assert np.linalg.norm(alpha-reference)/np.linalg.norm(reference)<=1e-6
    assert diagnostics['explicit_relative_residual']<1e-7 and len(diagnostics['history'])==1350


def test_standard_entry_point_full_native_output(tmp_path):
    destination=tmp_path/'nested/reconstruction.npz';alpha,_=run(ROOT/'data/raw_data.npz',destination)
    with np.load(destination,allow_pickle=False) as z:
        assert z.files==['reconstruction'];np.testing.assert_array_equal(z['reconstruction'],alpha[None])
        assert z['reconstruction'].dtype==np.float64 and z['reconstruction'].shape==(1,440,680)
