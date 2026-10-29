from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz',method='one_step'):
    image,mask,b,_=load_data(data_path);p=reconstruct(image,mask,b,method)
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,S0=p[None,...,0],diffusion=p[None,...,1],fraction=p[None,...,2],pseudodiffusion=p[None,...,3])
    return p

if __name__=='__main__':run()
