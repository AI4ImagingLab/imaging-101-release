from pathlib import Path
import json
import numpy as np
from .preprocessing import ROOT,load_observations
from .solvers import reconstruct
def main():
    result,info=reconstruct(*load_observations())
    out=ROOT/'output';out.mkdir(exist_ok=True)
    np.savez(out/'reconstruction.npz',**result)
    print(json.dumps(info))
    return result,info
if __name__=='__main__': main()
