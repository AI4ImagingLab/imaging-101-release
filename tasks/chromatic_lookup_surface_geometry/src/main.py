"""Standard python -m src.main entry; importing this module has no backend effect."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import solve

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz',block_size=2048):
    raw=load_observations(data_path);result=solve(raw,block_size=block_size)
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,reconstruction=result['reconstruction'][None])
    return result

def main():
    result=run()
    print(f"Reconstructed{result['reconstruction'].shape}; inverse seconds{result['seconds']:.6f}")
    return 0

if __name__=='__main__':raise SystemExit(main())
