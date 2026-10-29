"""Standard execution: python -m src.main."""
from pathlib import Path
import json,time
import numpy as np
from .preprocessing import load_observations,normalize_mosaic
from .solvers import reconstruct

def main():
    start=time.perf_counter();raw,black,_=load_observations()
    image=reconstruct(normalize_mosaic(raw,black))
    path=Path('output/reconstruction.npz');path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,reconstruction=image[None])
    print(json.dumps({'output':str(path),'shape':[1,*image.shape],'runtime_seconds':time.perf_counter()-start}))
    return image

if __name__=='__main__':main()
