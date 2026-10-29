"""Standard independently runnable full reconstruction entry point."""
from pathlib import Path
import time
import numpy as np
from .solvers import reconstruct

def main():
 """Read public observations and write the complete daily rainfall field, mm."""
 start=time.perf_counter()
 with np.load('data/raw_data.npz',allow_pickle=False) as raw:result=reconstruct(raw)
 Path('output').mkdir(exist_ok=True)
 np.savez_compressed('output/reconstruction.npz',reconstruction=result['reconstruction'])
 print(f"Complete daily field written; runtime {time.perf_counter()-start:.6f}s")

if __name__=='__main__':main()
