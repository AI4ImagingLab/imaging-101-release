"""Standard CLI. Importing this module does not select a plotting backend."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import estimate
def run(root: str | Path | None=None) -> dict:
    root=Path(__file__).resolve().parents[1] if root is None else Path(root)
    x=load_observations(root/'data/raw_data.npz');result=estimate(x['image'],x['multiple'],x['mask'])
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',**{key:result[key][None] for key in ('shading','reflectance')})
    return result
if __name__=='__main__':
    result=run();print('Fresh Weiss reconstruction:',result['integration'])
