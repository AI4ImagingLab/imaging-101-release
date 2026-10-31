"""Standard complete-observation reconstruction entry point."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct
ROOT=Path(__file__).resolve().parents[1]
def run(data_path=None,output_path=None):
    q,I,error=load_observations(ROOT/'data/raw_data.npz' if data_path is None else data_path)
    result=reconstruct(q,I,error)
    path=ROOT/'output/reconstruction.npz' if output_path is None else Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,reconstruction=result['reconstruction'][None],background=np.array([[result['background']]]))
    return result
if __name__=='__main__':
    result=run();print('iterations',result['iterations'],'chi_squared',float(result['residual']@result['residual']))
