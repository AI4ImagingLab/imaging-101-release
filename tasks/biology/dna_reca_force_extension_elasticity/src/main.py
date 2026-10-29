"""Standard task entry point, no plotting backend side effects."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct,submission
def main():
    data=load_data();fit=reconstruct(data)
    if not fit.success:raise RuntimeError(fit.message)
    Path('output').mkdir(exist_ok=True);np.savez_compressed('output/reconstruction.npz',**submission(fit,data))
if __name__=='__main__':main()
