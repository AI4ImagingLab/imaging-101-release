"""python -m src.main actively reconstructs every bound real homodyne sample."""
from pathlib import Path
import argparse,time
import numpy as np
from .preprocessing import load_observations,calibrate
from .solvers import maximum_likelihood
from .physics_model import wigner_function,Q_GRID,P_GRID

def main(diagnostics=None):
    start=time.perf_counter();root=Path(__file__).resolve().parents[1]
    data=load_observations();calibration=calibrate(data)
    result=maximum_likelihood(calibration['q_accepted'],calibration['phi_accepted'])
    result['wigner']=wigner_function(result['density_matrix'],Q_GRID,P_GRID)
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',density_matrix=result['density_matrix'][None],wigner=result['wigner'][None])
    if diagnostics is not None:
        path=Path(diagnostics);path.parent.mkdir(parents=True,exist_ok=True)
        arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        arrays.update({f'rho_step_{k}':v for k,v in result['snapshots'].items()})
        arrays.update({k:calibration[k] for k in ('phase_fit_param','cutoff','accepted_indices','q_accepted','phi_accepted')})
        arrays.update(iterations=np.array(result['iterations']),runtime_seconds=np.array(result['runtime_seconds']))
        np.savez_compressed(path,**arrays)
    print(f"Accepted {len(calibration['q_accepted'])} samples; {result['iterations']} complete ML updates; {time.perf_counter()-start:.6f}s",flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--diagnostics',type=Path)
    main(parser.parse_args().diagnostics)
