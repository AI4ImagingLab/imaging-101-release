from pathlib import Path
import numpy as np
from scipy import sparse
from src.physics_model import calibrate, integration_operators
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def fixture():return np.load(ROOT/'test/fixtures/qualified_upstream_stages.npz',allow_pickle=False)
def test_calibration_stage():
 f=fixture();out=calibrate(f['calibration__X_pix'],f['calibration__Y_pix'],f['calibration__U'],f['calibration__V'],f['calibration__sigma_U'],f['calibration__sigma_V'])
 for a,k in zip(out,['X','Y','rho_x','rho_y','sigma_rho_x','sigma_rho_y']):np.testing.assert_allclose(a,f['calibration__'+k],rtol=3e-14,atol=1e-13)
def test_operator_stage():
 f=fixture();D,B,C=integration_operators(f['calibration__X'],f['calibration__Y'])
 for a,k in [(D,'Operator_GLS'),(B,'Map_pgrad'),(C,'Map_dirichlet')]:
  prefix='wls__'+k; ref=sparse.csr_matrix((f[prefix+'__data'],f[prefix+'__indices'],f[prefix+'__indptr']),shape=tuple(f[prefix+'__shape']))
  np.testing.assert_allclose(a.toarray(),ref.toarray(),rtol=3e-14,atol=1e-13)
def test_full_qualified_pipeline():
 a=np.load(ROOT/'data/raw_data.npz',allow_pickle=False);out=reconstruct({k:a[k] for k in a.files})
 ref=np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False)
 for k in ['reconstruction','density_sigma']:np.testing.assert_allclose(out[k],ref[k],rtol=2e-11,atol=3e-12)
def test_poisson_comparison():
 a=np.load(ROOT/'data/raw_data.npz',allow_pickle=False);out=reconstruct({k:a[k] for k in a.files},method='poisson');f=fixture()
 np.testing.assert_allclose(out['reconstruction'][0],f['poisson__Pn_ex'][1:-1,1:-1],rtol=3e-13,atol=3e-12)
 np.testing.assert_allclose(out['density_sigma'][0],f['poisson__sigma_Pn_ex'][1:-1,1:-1],rtol=3e-12,atol=3e-12)
