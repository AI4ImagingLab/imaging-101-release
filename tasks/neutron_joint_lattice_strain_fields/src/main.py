"""Standard independent reference reconstruction entry point."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data,ROOT
from .solvers import reconstruct
def run(output=None):
    result,diagnostics=reconstruct(load_data());p=Path(output) if output else ROOT/'output/reconstruction.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**result);return result,diagnostics
if __name__=='__main__':
    result,info=run();print(f"{len(info['history'])-1} Newton updates, {info['elapsed']:.3f} seconds")
