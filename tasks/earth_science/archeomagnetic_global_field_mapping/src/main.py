"""Standard entry point; no backend change on import."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations,ROOT
from .solvers import reconstruct
from .physics_model import field_maps

def main():
    coefficients,diagnostics=reconstruct(load_observations())
    output=ROOT/'output/reconstruction.npz'; output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,reconstruction=field_maps(coefficients)[None],coefficients=coefficients[None])
    print(f'Saved {output.name}; final pre-update weighted MAE {diagnostics["weighted_mae"][-1]:.6f}')
    return output
if __name__=='__main__': main()
