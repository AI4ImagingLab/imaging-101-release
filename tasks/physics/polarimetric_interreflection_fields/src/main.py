"""Standard independent full-data entry point: python -m src.main."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def main():
    start=time.perf_counter();data=load_observations()
    result=reconstruct(data['intensity'],data['polarizer_psg'],data['polarizer_psa'])
    out=Path(__file__).resolve().parents[1]/'output';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'reconstruction.npz',reconstruction=result['reconstruction'][None])
    print(f"Complete16-state2048x2448 five-field reconstruction: {time.perf_counter()-start:.6f}s")
    return result

if __name__=='__main__':main()
