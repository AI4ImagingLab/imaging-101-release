"""Standard public-input reconstruction entry, no backend mutation on import."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    result,info=reconstruct(load_data(data_path))
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**result)
    return result,info

if __name__=='__main__':
    result,info=run()
    print('EM iterations:',info['iterations'],'scale:',result['scale'].item())
