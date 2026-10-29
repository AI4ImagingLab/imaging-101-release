"""Reproduce author binvox-based synthetic acquisition with explicit inputs.

The native asset identity/seed are maintainer provenance, not public inputs.
Read binvox RLE, canonical y rotation,10-voxel zero padding, eight uniform
true Euler triples, vonMises kappa100 uncertain readings, and opacity model.
No solver path calls generation or accesses its hidden object/true views.
"""
from pathlib import Path
import argparse
import numpy as np
import torch
from .physics_model import rotate_volume,project

def read_binvox(path):
    """Decode binary RLE and author fix_coords transpose(0,2,1)."""
    with Path(path).open('rb') as f:
        if not f.readline().startswith(b'#binvox'):raise ValueError('Not binvox')
        shape=None
        while True:
            line=f.readline()
            if not line:raise ValueError('Missing data marker')
            if line.startswith(b'dim'):shape=tuple(map(int,line.split()[1:]))
            if line.strip()==b'data':break
        pairs=np.frombuffer(f.read(),dtype=np.uint8)
    if shape is None or len(pairs)%2:raise ValueError('Malformed binvox')
    values=np.repeat(pairs[::2],pairs[1::2].astype(np.int64))
    if len(values)!=np.prod(shape):raise ValueError('RLE length mismatch')
    return values.reshape(shape).astype(np.float32).transpose(0,2,1).copy()

def generate(object_path,raw_output,truth_output,seed,device='cuda'):
    """Generate observations/truth; seed explicit and excluded from raw output."""
    random=np.random.RandomState(seed)
    true_views=torch.tensor(random.rand(8,3)*2*np.pi,dtype=torch.float32)
    prior=torch.tensor(random.vonmises(true_views.numpy(),100.,size=(8,3)),dtype=torch.float32)
    volume=torch.tensor(read_binvox(object_path),device=device)
    with torch.no_grad():
        volume=rotate_volume(volume,torch.tensor([0.,np.pi/2.,0.],dtype=torch.float32))
        volume=torch.nn.functional.pad(volume,(10,10,10,10,10,10))
        images=project(volume,true_views)
    for path in [raw_output,truth_output]:Path(path).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(raw_output,silhouettes=images.cpu().numpy().transpose(2,0,1)[None],view_prior=prior.numpy()[None])
    np.savez_compressed(truth_output,reconstruction=volume.cpu().numpy()[None],viewpoints=true_views.numpy()[None])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--object',required=True);p.add_argument('--raw-output',required=True);p.add_argument('--truth-output',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(1);generate(a.object,a.raw_output,a.truth_output,a.seed,a.device)
