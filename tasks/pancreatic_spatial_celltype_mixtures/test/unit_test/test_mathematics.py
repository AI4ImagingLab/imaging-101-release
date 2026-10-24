import importlib.util
from pathlib import Path
import numpy as np
import pytest
from src.physics_model import forward_model,car_precision,log_objective,normalize_weights
from src.preprocessing import reference_basis
from src.generate_data import thin_counts
ROOT=Path(__file__).resolve().parents[2]

def test_linear_forward_adjoint_identity():
    rng=np.random.RandomState(4);B=rng.rand(9,3);V=rng.rand(7,3);Y=rng.rand(9,7)
    assert np.sum(forward_model(B,V)*Y)==pytest.approx(np.sum(V*(Y.T@B)),rel=2e-15)
def test_reference_basis_analytic_independent_cells():
    C=np.array([[1.,3.,2.,4.],[3.,1.,6.,2.]])
    B,active=reference_basis(C,np.array([0,0,1,1]),np.zeros(4),2)
    np.testing.assert_allclose(B,[[2,3],[2,4]],atol=2e-15)
    assert active.all()
def test_car_positive_definite_and_scale():
    W=np.array([[0.,1.,.2],[1.,0.,.4],[.2,.4,0.]])
    L=car_precision(W,.9)
    assert np.linalg.eigvalsh(L).min()>0
    np.testing.assert_allclose(L@np.ones(3),.1*W.sum(1),atol=3e-16)
    with pytest.raises(ValueError):car_precision(W,1.)
def test_simplex_analytic():
    np.testing.assert_array_equal(normalize_weights(np.array([[2.,6.],[3.,3.]])),[[.25,.75],[.5,.5]])
    with pytest.raises(ValueError):normalize_weights(np.zeros((2,2)))
def test_direct_objective_independent_scalar():
    X=np.array([[2.,3.]]);B=np.array([[1.]])
    V=np.array([[1.],[2.]]);W=np.array([[0.,1.],[1.,0.]])
    # SSE=2, variance=2; Z=(.5,1.5), quadratic=1.75; lambda=2, beta=1.
    val=log_objective(X,B,V,np.array([.5]),2.,np.array([2.]),.5,W)
    expected=-np.log(2)-.5-np.log(2)-1.75/4-2*np.log(2)-.5
    assert val==pytest.approx(expected,abs=2e-15)
def test_thinning_exact_seed_and_units():
    C=np.array([[0,1,10],[20,100,1000]],np.int32)
    np.testing.assert_array_equal(thin_counts(C,.9,42),[[0,0,9],[18,93,896]])
    np.testing.assert_array_equal(thin_counts(C,1.,42),C)
    assert thin_counts(C,.9,42).dtype==np.int32
@pytest.mark.parametrize("C,p",[(np.ones(3),.5),(np.array([-1]),.5),(np.array([1]),0),(np.array([1]),1.1)])
def test_thinning_invalid(C,p):
    with pytest.raises(ValueError):thin_counts(C,p,0)
def test_runtime_no_domain_packages():
    import ast
    for path in (ROOT/"src").glob("*.py"):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names=[x.name.split('.')[0] for x in node.names]
            elif isinstance(node,ast.ImportFrom):names=[(node.module or '').split('.')[0]]
            else:continue
            assert all(n in {'','pathlib','time','numpy','scipy','matplotlib','textwrap','physics_model','preprocessing','solvers'} for n in names)


def test_gaussian_normalizer_independent_eigenvalues():
    from src.physics_model import car_log_normalizer
    W=np.array([[0.,1.],[1.,0.]])
    # L has eigenvalues1-phi and1+phi, determinant1-phi?.
    assert car_log_normalizer(W,.5,3)==pytest.approx(1.5*np.log(.75),abs=4e-16)
    # Dropping this quantity is harmless only when phi is fixed.
    assert car_log_normalizer(W,.9,3)!=car_log_normalizer(W,.5,3)
