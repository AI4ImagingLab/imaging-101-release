from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    root=Path(__file__).resolve().parents[1];trace=load_data(root/'data/raw_data.npz')
    result=reconstruct(trace);(root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=result['parameters'][None])
    print(result['parameters'])
if __name__=='__main__':main()
