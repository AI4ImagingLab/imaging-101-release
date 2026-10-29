import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from src.preprocessing import load_data
from src.solvers import (prepare_inverse,tv_value_gradient_hessian,Objective,
                         gauss_newton_direction,uniform_line_search)

ROOT=Path(__file__).resolve().parents[2]

def context():return prepare_inverse(load_data(ROOT/'data/raw_data.npz'))

def test_source_initialization_geometry_TV_and_gaussian_precision():
    c=context()
    with np.load(ROOT/'test/fixtures/upstream_initial_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(c.initial_modulus,f['output_initial_modulus'])
        assert abs(c.alpha-f['output_alpha'].item())<1e-17
        assert abs(c.beta-f['output_beta'].item())<1e-12
        assert np.all(f['output_face_weights']==1)
        assert np.linalg.norm(c.gaussian_precision-f['output_precision'])/np.linalg.norm(f['output_precision'])<1e-10
        R=csr_matrix((f['output_R_data'],f['output_R_indices'],f['output_R_indptr']),shape=tuple(f['output_R_shape']))
        count=R.shape[0]//2
        source_faces={tuple(sorted(set(R[i].indices)|set(R[i+count].indices))):i for i in range(count)}
        assert len(source_faces)==1521
        order=np.array([source_faces[tuple(sorted(row))] for row in c.dual_triangles])
        assert np.max(abs((c.gradient_x-R[order]).toarray()))<2e-14
        assert np.max(abs((c.gradient_y-R[order+count]).toarray()))<2e-14
        _,gradient,H=tv_value_gradient_hessian(c.initial_modulus,c.gradient_x,c.gradient_y,c.beta)
        np.testing.assert_allclose(gradient,f['output_TV_gradient_initial'],rtol=0,atol=1e-10)
        assert np.linalg.norm(H-f['output_TV_Hessian_initial'])/np.linalg.norm(f['output_TV_Hessian_initial'])<1e-10

def test_complete_current_Jacobian_and_GN_vs_qualified_source():
    c=context();objective=Objective(c)
    tolerance=json.loads((ROOT/'test/fixtures/stage_tolerances.json').read_text())
    with np.load(ROOT/'test/fixtures/upstream_initial_stages.npz',allow_pickle=False) as f:
        J=c.model.jacobian(c.initial_modulus)
        assert np.linalg.norm(J-f['output_J_initial'])/np.linalg.norm(f['output_J_initial'])<tolerance['frozen_J_relative_frobenius_boundary']
        prediction=c.model.forward(c.initial_modulus)
        direction=gauss_newton_direction(c.initial_modulus,prediction,J,objective)
        assert np.linalg.norm(direction-f['output_direction_initial'])/np.linalg.norm(f['output_direction_initial'])<tolerance['frozen_GN_direction_relative_boundary']

def test_entire_source_line_search_grids_and_objectives():
    c=context();objective=Objective(c)
    with np.load(ROOT/'test/fixtures/upstream_first_line_search.npz',allow_pickle=False) as f:
        step,coarse,cost,refined,refined_cost=uniform_line_search(f['input_modulus'],f['input_direction'],objective,True)
        np.testing.assert_array_equal(coarse,f['output_coarse_steps'])
        np.testing.assert_array_equal(refined,f['output_refined_steps'])
        assert step==f['output_step'].item()
        np.testing.assert_allclose(cost,f['output_coarse_costs'],rtol=1e-9,atol=1e-7)
        np.testing.assert_allclose(refined_cost,f['output_refined_costs'],rtol=1e-9,atol=1e-7)
