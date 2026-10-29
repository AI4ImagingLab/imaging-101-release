"""Standard CLI: python -m src.main. No backend mutation on import."""
from pathlib import Path
import numpy as np
from .solvers import reconstruct
def main():
    root=Path(__file__).resolve().parents[1]
    with np.load(root/'data/raw_data.npz',allow_pickle=False) as f:raw={k:f[k] for k in f.files}
    out=reconstruct(raw);target=root/'output/reconstruction.npz';target.parent.mkdir(exist_ok=True)
    np.savez_compressed(target,reconstruction=out['reconstruction'])
    print('Computed',out['iterations'],'iterations; wrote',target)
if __name__=='__main__':main()
