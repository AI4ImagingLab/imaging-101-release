from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from src.preprocessing import load_data
from src.physics_model import derivative_matrices
from src.solvers import integrate,reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_normal_decoding_and_all_original_operators():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        for name in ['vase','tent']:
            n,m=data[name]
            np.testing.assert_array_equal(n,f[name+'_input_normals'])
            np.testing.assert_array_equal(m,f[name+'_input_mask'])
            encoded=f[name+'_input_encoded'];levels=65535 if encoded.dtype==np.uint16 else 255
            np.testing.assert_array_equal(n,encoded/levels*2-1)
            for tag,a in zip(['right','left','up','down'],derivative_matrices(m,-n[m,2],-n[m,2])):
                original=csr_matrix((f[name+'_output_'+tag+'_data'],f[name+'_output_'+tag+'_indices'],f[name+'_output_'+tag+'_indptr']),shape=a.shape)
                difference=a-original;assert difference.nnz==0

def test_original_inverse_updates_and_full_results():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        for name,k in [('vase',4.),('tent',1.)]:
            n,m=data[name];z,diag=integrate(n,m,k=k)
            np.testing.assert_allclose(z,f[name+'_output_depth'],rtol=1e-9,atol=1e-8)
            np.testing.assert_allclose(diag['energy'],f[name+'_output_energy'],rtol=1e-9,atol=1e-8)
            np.testing.assert_allclose(diag['horizontal_weights'],f[name+'_output_wu'],rtol=1e-9,atol=1e-8)
            np.testing.assert_allclose(diag['vertical_weights'],f[name+'_output_wv'],rtol=1e-9,atol=1e-8)
    output,_=reconstruct(data)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        for k in f.files:np.testing.assert_allclose(output[k],f[k],rtol=1e-9,atol=1e-8)
