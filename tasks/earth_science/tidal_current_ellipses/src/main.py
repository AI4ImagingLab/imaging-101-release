from pathlib import Path
import numpy as np
from .preprocessing import load_data,polar_to_velocity
from .solvers import solve
def reconstruct(path='data/raw_data.npz'):
    a=load_data(path);v=polar_to_velocity(a['speed'],a['direction_deg'])
    return solve(a['time_seconds'],v,a['frequency_cph'])
def main():
    result=reconstruct();Path('output').mkdir(exist_ok=True);np.savez_compressed('output/reconstruction.npz',**result)
if __name__=='__main__':main()
