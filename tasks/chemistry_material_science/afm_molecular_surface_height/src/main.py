"""Standard CLI: python -m src.main. Importing this module leaves backend alone."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def main():
    measurements,probe=load_observations()
    surfaces=reconstruct(measurements,probe)
    Path("output").mkdir(exist_ok=True)
    np.savez_compressed("output/reconstruction.npz",reconstruction=surfaces[None])
    return surfaces

if __name__ == "__main__":
    main()
