"""Independent mathematical checks: no target pose/GT/solver cache inputs."""
import numpy as np
import pytest
from src.preprocessing import compute_attributes,compute_mgc,load_data
from src.physics_model import (ROOTS,reorder_scores,round_angles,connect_graph,component_labels,
                   rotation_step,legacy_cost,find_mismatches,four_loops)


def analytic_tiles(n=6,p=5):
    x=np.arange(n*p*p*3,dtype=float).reshape(n,p,p,3)
    return ((np.sin(x*.37)+np.cos(x*.131))*700+1400).astype(np.uint16)


def test_public_loader_preserves_precision(tmp_path):
    tiles=analytic_tiles();path=tmp_path/'raw.npz';np.savez(path,tiles=tiles[None])
    assert np.array_equal(load_data(path),tiles)
    np.savez(path,tiles=tiles[None].astype(np.float32))
    with pytest.raises(ValueError):load_data(path)


def test_edge_conventions_and_covariance_positive():
    tiles=analytic_tiles();attrs=compute_attributes(tiles)
    for a in attrs:
        assert a['pix'].dtype==a['D_mu'].dtype==a['D_cov'].dtype==np.float64
        assert np.array_equal(a['pixR'].reshape(6,3,5),a['pix'].reshape(6,3,5)[:,:,::-1])
        assert np.min(np.linalg.eigvalsh(a['D_cov'].transpose(2,0,1)))>0
    assert np.array_equal(attrs[0]['pix'][0].reshape(3,5).T,tiles[0,0])
    assert np.array_equal(attrs[2]['pix'][0].reshape(3,5).T,tiles[0,-1,::-1])
    const=np.full((1,5,5,3),12,np.uint16);a=compute_attributes(const)[0]
    # Nine dummy samples have zero mean and RGB Gram [[4,2,2],...].
    np.testing.assert_array_equal(a['D_mu'],np.zeros((1,3)))
    np.testing.assert_allclose(a['D_cov'][:,:,0],np.array([[4,2,2],[2,4,2],[2,2,4]])/13,rtol=0,atol=0)


def test_all_sixteen_reorder_actions_are_physical_rotations():
    tiles=analytic_tiles();scores=compute_mgc(tiles)
    for ri in range(4):
        for rj in range(4):
            nums=np.array([ri,rj,0,1,2,3])
            changed=np.stack([np.rot90(t,-int(r)) for t,r in zip(tiles,nums)])
            fresh=compute_mgc(changed)
            # May differ by float32 roundoff due source single gradient means.
            np.testing.assert_allclose(reorder_scores(scores,nums),fresh,rtol=2*np.finfo(np.float32).eps,atol=0)
    assert np.isposinf(scores[np.arange(6),np.arange(6)]).all()


def test_connection_random_ties_structure_not_seed_identity():
    n=8;weights=np.ones((n,n));np.fill_diagonal(weights,np.inf)
    nn=np.zeros((n,n));nn[0,1]=nn[1,0]=1
    # No fixed RNG or expected stochastic trajectory. Every valid tie outcome
    # must preserve prior edges, symmetry, degree positivity and connectivity.
    for _ in range(8):
        connected=connect_graph(nn,weights)
        assert np.all(connected==connected.T) and connected[0,1]==1
        assert len(np.unique(component_labels(connected)))==1
        assert np.count_nonzero(np.triu(connected==.001,1))==6


def test_consistent_c4_left_singular_operator_and_gauge():
    r=np.array([0,1,2,3,1,0]);z=ROOTS[r];rot=z[:,None]*z.conj()[None,:]
    nn=np.ones((6,6))-np.eye(6);tiles=analytic_tiles()
    new,nums,vector,A=rotation_step(nn,rot,tiles)
    np.testing.assert_allclose(A@z,0,atol=1e-15)
    phase=vector/z;np.testing.assert_allclose(phase,phase[0],atol=1e-15)
    assert len(np.unique((nums-r)%4))==1
    for i in range(6):assert np.array_equal(new[i],np.rot90(tiles[i],-int(nums[i])))


def test_source_rounding_at_midpoints():
    # Exact equal-component complex vectors avoid exp() perturbing a tie.
    assert np.array_equal(round_angles(np.array([1+1j,-1+1j,-1-1j,1-1j])),[1,2,3,0])
    assert np.array_equal(round_angles(ROOTS),[0,1,2,3])


def test_legacy_cost_explicit_biased_boundary():
    goal=np.arange(1,7).reshape(2,3)
    scores=np.ones((6,6,4),np.float32)
    # Left:4, top:3, right:2 (last-column edge omitted), bottom:0.
    assert legacy_cost(scores,goal)==9


def test_four_loop_consistent_square_adds_diagonal_relations():
    nn=np.zeros((4,4));
    for i,j in [(0,1),(1,2),(2,3),(3,0)]:nn[i,j]=nn[j,i]=1
    z=ROOTS[np.array([0,1,3,2])];rot=z[:,None]*z.conj()[None,:]
    out,r,d=four_loops(nn,rot)
    assert d[0,2]==d[2,0]==d[1,3]==d[3,1]==1
    np.testing.assert_array_equal(r,rot)
    np.testing.assert_array_equal(out,nn)
