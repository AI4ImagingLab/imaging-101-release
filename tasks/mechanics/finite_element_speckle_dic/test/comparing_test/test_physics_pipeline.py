from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
from src.physics_model import prepare_reference, filtered_frames, sample_element

ROOT=Path(__file__).resolve().parents[2]


def test_measured_observation_consistency():
    reference,deformed,nodes,elements=load_data(ROOT/'data/raw_data.npz')
    displacement,_=reconstruct(reference,deformed,nodes,elements)
    blocks,_=prepare_reference(reference,nodes,elements)
    frames=filtered_frames(deformed,nodes,elements)
    def residual(current):
        arrays=[I0-sample_element(frame,lo,N,current[e])
                for e,(N,B,I0),(frame,lo) in zip(elements,blocks,frames)]
        return np.sqrt(np.mean(np.concatenate(arrays)**2))
    # Independent original-code acceptance: improve mismatch by at least 50%.
    assert residual(nodes+displacement)<.5*residual(nodes)
    for e,(N,B,I0),(frame,lo) in zip(elements,blocks,frames):
        mapped=N@(nodes[e]+displacement[e]-lo+10)
        assert mapped.min()>4
        assert np.all(mapped[:,0]<frame.shape[1]-4)
        assert np.all(mapped[:,1]<frame.shape[0]-4)
