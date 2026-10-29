from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
from src.preprocessing import load_observations,decompose
from src.physics_model import diffuse_zenith,fit_light,angular_fields,assemble_height_system
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def prepared():
    arrays=load_observations(ROOT/'data/raw_data.npz')
    fixture=np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)
    return arrays,fixture


def test_original_preprocessing_fresnel_and_light(prepared):
    (images,angles,mask,spec),f=prepared
    rho,phase,intensity=decompose(images,angles,mask)
    for value,key in [(rho,'rho'),(phase,'phi'),(intensity,'intensity')]:
        np.testing.assert_allclose(value,f['output_'+key],atol=2e-12,rtol=1e-10)
    light,_,iterations=fit_light(diffuse_zenith(rho),phase,intensity,mask&~spec)
    np.testing.assert_allclose(light,f['output_light'],atol=2e-13,rtol=1e-11)
    assert iterations==9
    theta,adjusted=angular_fields(rho,phase,spec)
    np.testing.assert_allclose(theta,f['output_theta'],atol=2e-12,rtol=1e-10)
    np.testing.assert_allclose(adjusted,f['output_phase'],atol=2e-12,rtol=1e-10)


def test_original_full_sparse_operator_and_rhs(prepared):
    (_,_,mask,spec),f=prepared
    C,d=assemble_height_system(f['output_theta'],np.minimum(1,f['output_intensity']),
                              f['output_phase'],f['output_light'],mask,spec)
    expected=sparse.coo_matrix((f['output_Cdata'],(f['output_Crow'],f['output_Ccol'])),
                              shape=tuple(f['output_Cshape'])).tocsr()
    assert C.shape==expected.shape and C.nnz==expected.nnz
    np.testing.assert_array_equal(C.indptr,expected.indptr)
    np.testing.assert_array_equal(C.indices,expected.indices)
    np.testing.assert_allclose(C.data,expected.data,atol=2e-14,rtol=1e-12)
    np.testing.assert_allclose(d,f['output_rhs'],atol=2e-13,rtol=1e-12)


def test_original_full_pipeline_and_normal_equations(prepared):
    arrays,f=prepared
    height,info=reconstruct(*arrays)
    np.testing.assert_allclose(height,f['output_height'],atol=2e-6,rtol=1e-7)
    assert np.linalg.norm(info['normal_residual'])<1e-8
    assert np.isfinite(height).all()
    assert np.max(np.abs(height[~arrays[2]]))==0


def test_standard_entry_point_and_fresh_judge(tmp_path):
    from src.main import run
    import importlib.util
    target=tmp_path/'reconstruction.npz'
    height,_,seconds=run(ROOT/'data/raw_data.npz',target)
    assert seconds>0 and height.shape==(457,235)
    spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    report,code=judge.evaluate(target)
    assert code==0 and report['passed']
