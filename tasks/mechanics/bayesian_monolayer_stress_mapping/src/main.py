"""Standard active reconstruction entry point."""
from pathlib import Path
import time,numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    start=time.perf_counter();stress,state=reconstruct(load_data())
    Path("output").mkdir(exist_ok=True);np.savez_compressed("output/reconstruction.npz",reconstruction=stress[None])
    print({"seconds":time.perf_counter()-start,"traction_rms_kpa":float(np.sqrt(np.mean((state["prediction"]-state["traction"].reshape(2,50,50))**2))),"gradient_norm":float(np.linalg.norm(state["gradient"]))})
    return stress,state
if __name__=="__main__":main()
