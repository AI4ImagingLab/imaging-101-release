"""Fixed-reference Gauss-Newton Q4 DIC; no domain-library calls."""
import numpy as np
from .physics_model import prepare_reference, filtered_frames, sample_element


def reconstruct(reference, deformed, nodes, elements, max_iterations=80,
                tolerance=1e-8, order=4, pad=10):
    """Return nodal displacement (N,2) and numerical iteration diagnostics."""
    blocks, inverse = prepare_reference(reference, nodes, elements)
    frames = filtered_frames(deformed, nodes, elements, pad, order)
    current = nodes.copy()
    history = []
    for iteration in range(max_iterations):
        rhs = np.zeros(2*len(nodes))
        loss = 0.
        for element, (N, B, I0), (frame, lower) in zip(elements, blocks, frames):
            residual = I0-sample_element(frame, lower, N, current[element], pad, order)
            contribution = B @ residual
            rhs[2*element] += contribution[:4]
            rhs[2*element+1] += contribution[4:]
            loss += float(residual @ residual)
        increment = inverse @ rhs
        current += increment.reshape(-1, 2)
        history.append((loss, float(np.max(np.abs(increment)))))
        if history[-1][1] < tolerance:
            return current-nodes, np.array(history)
    raise RuntimeError('Q4 DIC did not meet the declared convergence tolerance')
