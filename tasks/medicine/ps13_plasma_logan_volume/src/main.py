"""Reconstruct the complete bound native PET slice."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    start=time.perf_counter();activity,mask,plasma,frames,durations,_=load_data()
    volume,offset=reconstruct(activity,mask,plasma,frames,durations)
    Path("output").mkdir(exist_ok=True);np.savez_compressed("output/reconstruction.npz",reconstruction=volume[None])
    print(f"Reconstructed {int(mask.sum())} voxel distribution volumes in {time.perf_counter()-start:.6f}s")
    return volume,offset

if __name__=="__main__":main()
