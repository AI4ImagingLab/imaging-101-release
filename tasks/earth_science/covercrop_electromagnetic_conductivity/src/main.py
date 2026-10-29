import json
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    data=load_data();image,diagnostics=reconstruct(data)
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=image[None])
    print(json.dumps({'shape':list(image.shape),'optimizer_successes':sum(x['success'] for x in diagnostics)},allow_nan=False))
if __name__=='__main__': main()
