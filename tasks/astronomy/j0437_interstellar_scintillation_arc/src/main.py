"""Standard complete real-data reconstruction entry point."""
from pathlib import Path
import json,numpy as np
from .solvers import reconstruct

def run(output_path='output/reconstruction.npz'):
    result,diagnostics,record=reconstruct()
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,reconstruction=result)
    print(json.dumps({**record,'kappa_s2_per_m':result.tolist()}))
    return result

if __name__=='__main__':run()
