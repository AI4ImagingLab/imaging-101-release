"""Standard fresh reconstruction entry point."""
from pathlib import Path
import json
import numpy as np
from .preprocessing import load_data,ROOT
from .solvers import reconstruct,submission

def main():
    vector,diagnostics=reconstruct(load_data())
    out=ROOT/'output';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'reconstruction.npz',**submission(vector))
    print(json.dumps(diagnostics,indent=2))
if __name__=='__main__':main()
