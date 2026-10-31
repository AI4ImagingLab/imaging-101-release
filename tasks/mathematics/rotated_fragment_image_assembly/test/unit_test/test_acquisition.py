from pathlib import Path
import numpy as np
from src.generate_data import acquire,adjoint
from src.main import goal_to_positions


def test_pixel_permutation_adjoint_and_energy():
    # Explicit independent permutation, every rotation represented.
    image=np.arange(4*6*3,dtype=float).reshape(4,6,3)/7
    p=np.array([[1,2],[0,1],[1,0],[0,2],[0,0],[1,1]])
    q=np.array([0,1,2,3,1,2]);y=acquire(image,p,q,2)
    np.testing.assert_array_equal(adjoint(y,p,q),image)
    probe=np.cos(np.arange(y.size)).reshape(y.shape)
    np.testing.assert_allclose(np.vdot(y,probe),np.vdot(image,adjoint(probe,p,q)),rtol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(y),np.linalg.norm(image),rtol=1e-14)


def test_bound_forward_roundtrip_keeps_uint16():
    task=Path(__file__).resolve().parents[2]
    with np.load(task/'data/raw_data.npz',allow_pickle=False) as z:tiles=z['tiles'][0]
    with np.load(task/'evaluation/ground_truth.npz',allow_pickle=False) as z:p=z['positions'][0];q=z['quarter_turns'][0]
    image=adjoint(tiles,p,q)
    assert image.dtype==np.uint16 and image.shape==(504,672,3)
    np.testing.assert_array_equal(acquire(image,p,q),tiles)


def test_goal_index_order_and_malformed_assembly():
    import pytest
    np.testing.assert_array_equal(goal_to_positions([[3,1],[4,2]],4),[[0,1],[1,1],[0,0],[1,0]])
    for g in ([[1,1],[3,4]],[[0,2],[3,4]],[[1,2,3]]):
        with pytest.raises(ValueError):goal_to_positions(g,4)
