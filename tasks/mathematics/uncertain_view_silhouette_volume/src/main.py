"""Standard full reconstruction entry point."""
from pathlib import Path
import argparse
import os
import numpy as np
import torch
from .preprocessing import load_observations
from .solvers import reconstruct

def run(input_path='data/raw_data.npz',output_path='output/reconstruction.npz',device='cuda',callback=None):
    """Run selected full500-step method; save density and view output arrays."""
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Full benchmark requires tested CUDA runtime; CPU full pipeline unverified')
    if device=='cuda':
        free,total=torch.cuda.mem_get_info();budget=free-2*1024**3
        if budget<6*1024**3:raise RuntimeError('Insufficient GPU memory with2GiB safety headroom')
        torch.cuda.set_per_process_memory_fraction(budget/total)
    volume,views,history,diagnostics=reconstruct(*load_observations(input_path),device=device,callback=callback)
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,reconstruction=volume[None],viewpoints=views[None])
    return volume,views,history,diagnostics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='data/raw_data.npz');parser.add_argument('--output',default='output/reconstruction.npz');a=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    _,_,history,info=run(a.input,a.output,callback=lambda n,v:print(f'{n}/500 opacityMAE={v[0]:.8f}',flush=True) if n%25==0 else None)
    print(f'Saved {a.output};runtime={info["runtime_seconds"]:.3f}s;final dataMAE={history[-1,0]:.8f}')
if __name__=='__main__':main()
