from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from src import solvers as p
D=Path(__file__).resolve().parents[1]/'fixtures/position'
def fixture(name):return np.load(D/name,allow_pickle=False)
def sparse(z,k):return csr_matrix((z[k+'_data'],z[k+'_indices'],z[k+'_indptr']),shape=tuple(z[k+'_shape']))
def test_source_normalization_and_weights():
 with fixture('source_scores.npz') as z:s=z['SCO']
 with fixture('source_normalization.npz') as z:
  actual=p.normalize_scores(s);assert np.array_equal(np.isnan(actual),np.isnan(z['normSCO']));assert np.array_equal(np.isinf(actual),np.isinf(z['normSCO']));np.testing.assert_allclose(actual,z['normSCO'],rtol=2e-6,atol=2e-6,equal_nan=True)
  np.testing.assert_allclose(p.candidate_matches(s),z['info'],rtol=2e-6,atol=2e-6)
def test_original_lp_matrices_and_feasibility():
 with fixture('source_lp.npz') as z:
  f,A,b,E,d=p.build_lp(432,z['info'],z['fix_id'],z['fix_value'])
  np.testing.assert_allclose(f,z['f'],rtol=2e-6,atol=0)
  assert A.shape==sparse(z,'A').shape and E.shape==sparse(z,'Aeq').shape
  assert (A-sparse(z,'A')).nnz==0 and (E-sparse(z,'Aeq')).nnz==0
  np.testing.assert_array_equal(b,z['b']);np.testing.assert_array_equal(d,z['beq'])
  x,diag=p.solve_lp(f,A,b,E,d);assert diag['equality_max']<2e-6 and diag['inequality_max']<2e-6
  assert abs(f@x-f@z['x'])<2e-6*max(1,abs(f@z['x']))
def test_source_components():
 with fixture('source_lp.npz') as z:x=z['x'];info=z['info']
 labels,blocks,kept=p.recover_components(x,info,432)
 with fixture('source_geometry.npz') as z:
  assert np.array_equal(labels[:,None]==labels[None,:],z['labels'][:,None]==z['labels'][None,:]);assert len(blocks)==1;np.testing.assert_array_equal(blocks[0],z['initial_goal']);np.testing.assert_array_equal(kept,z['kept'])
def test_join_trim_greedy_and_fill_original_controls():
 with fixture('source_controls.npz') as z:
  joined,ok=p.join_blocks(np.array([[1],[2]]),np.array([[3],[4]]),1,3,2);assert ok;np.testing.assert_array_equal(joined,z['joined'])
  np.testing.assert_array_equal(p.trim_block(z['big'],2,3),z['trimmed'])
  np.testing.assert_array_equal(p.greedy_assemble([np.array([[1],[2]]),np.array([[3],[4]]),np.array([[5],[6]])],p.normalize_scores(z['scores'])),z['greedy_goal'])
  result=p.fill_holes(z['holey'],z['scores'],6)
 with fixture('source_fill_predelete.npz') as z:np.testing.assert_array_equal(result,z['b'])
def test_full_bound_source_candidate_geometry():
 with fixture('source_scores.npz') as z:goal,trace=p.assemble_positions(z['SCO'])
 with fixture('source_geometry.npz') as z:np.testing.assert_array_equal(goal,z['final_goal'])
 assert trace['lp_calls']>=1 and trace['inner_iterations_executed']<=5
 assert set(goal.ravel())==set(range(1,433))

def test_lp_analytic_zero_objective_and_free_negative_anchor():
 with fixture('source_controls.npz') as z:
  grid=z['grid'];clean_matches=[]
  for r in range(2):
   for c in range(3):
    for side,(dr,dc) in enumerate([(-1,0),(0,1),(1,0),(0,-1)],1):
     if 0<=r+dr<2 and 0<=c+dc<3:clean_matches.append([grid[r,c],grid[r+dr,c+dc],1.,side])
  f,A,b,E,d=p.build_lp(6,np.asarray(clean_matches),np.array([1,7]),np.array([-100000.,-100000.]))
  xy=np.zeros((6,2))
  for r in range(grid.shape[0]):
   for c in range(grid.shape[1]):xy[int(grid[r,c])-1]=[r,c]
  anchor=0;xy=xy-xy[anchor]-100000
  candidate=np.r_[xy[:,0],xy[:,1],np.zeros(len(f)-12)]
  np.testing.assert_allclose(E@candidate,d,atol=1e-12);assert (A@candidate<=b+1e-12).all();assert f@candidate==0
  x,diag=p.solve_lp(f,A,b,E,d);assert abs(diag['objective'])<1e-10;assert x[anchor]<0

def test_collision_rejection_and_directional_occupied_seams():
 a=np.array([[1,2]]);b=np.array([[2,3]]);result,valid=p.join_blocks(a,b,1,2,2);assert not valid;np.testing.assert_array_equal(result,b)
 scores=np.ones((6,6,4),dtype=np.float32);grid=np.array([[1,3],[2,4]]);labels=np.array([1,1,1,1,-1,-1])
 changed=p.update_scores(scores,[grid],labels)
 assert np.isfinite(changed[0,4,0])  # open top of piece1
 assert np.isinf(changed[0,4,2])  # bottom already occupied by piece2
 assert np.isinf(changed[4,0,0])  # reciprocal seam is also unavailable
