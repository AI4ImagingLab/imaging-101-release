from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import prepare_reference
from src.solvers import reconstruct

ROOT = Path(__file__).resolve().parents[2]


def test_original_stages_and_pipeline():
    image, deformed, nodes, elements = load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz', allow_pickle=False) as oracle:
        blocks, inverse = prepare_reference(image, nodes, elements)
        np.testing.assert_allclose(inverse, oracle['output_inverse_hessian'], rtol=1e-10, atol=1e-9)
        for e, (N, B, I0) in enumerate(blocks):
            for name, value in [('N', N), ('B', B), ('I', I0)]:
                np.testing.assert_allclose(value, oracle['output_'+name+str(e)], rtol=1e-10, atol=1e-9)
        displacement, history = reconstruct(image, deformed, nodes, elements)
        np.testing.assert_allclose(displacement, oracle['output_displacement'], atol=1e-8, rtol=0)
        assert len(history) == 10
        assert history[-1, 1] < 1e-8


def test_first_update():
    from src.physics_model import filtered_frames, sample_element
    image, deformed, nodes, elements = load_data(ROOT/'data/raw_data.npz')
    blocks, inverse = prepare_reference(image, nodes, elements)
    rhs = np.zeros(2*len(nodes))
    for e, (N, B, I0), (frame, lo) in zip(elements, blocks, filtered_frames(deformed, nodes, elements)):
        contribution = B @ (I0-sample_element(frame, lo, N, nodes[e]))
        rhs[2*e] += contribution[:4]
        rhs[2*e+1] += contribution[4:]
    with np.load(ROOT/'test/fixtures/upstream_stages.npz', allow_pickle=False) as data:
        np.testing.assert_allclose(nodes+(inverse@rhs).reshape(-1,2), data['output_first_nodes'], atol=1e-8, rtol=0)
