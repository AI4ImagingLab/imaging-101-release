"""python -m src.main writes the four declared reconstruction arrays."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct
def main():
    root=Path(__file__).resolve().parents[1]
    a,b=load_observations(root/'data/raw_data.npz');result=reconstruct(a,b)
    output=root/'output/reconstruction.npz';output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**{k:result[k][None].astype(np.float64) for k in ['phase_B','phase_E','Bx','By']})
    return output
if __name__=='__main__':print(main())
