"""Standard entry point: python -m src.main."""
from pathlib import Path
import numpy as np
from .solvers import reconstruct

def load_observations(path):
    """Read finite batch-first raw uint16 STCYX and fixed boolean support."""
    with np.load(path,allow_pickle=False) as a:
        if set(a.files)!={'observations','observation_mask'}:raise ValueError('unexpected observation keys')
        raw=a['observations'];mask=a['observation_mask']
    if raw.shape!=(1,3,2,2,512,512) or raw.dtype!=np.uint16:raise ValueError('invalid raw observations')
    if mask.shape!=(1,512,512) or mask.dtype!=np.bool_ or not mask.any():raise ValueError('invalid observation support')
    return raw[0],mask

def run(task_dir=None):
    """Reconstruct task input and write both numeric fields in the output NPZ."""
    root=Path(task_dir) if task_dir is not None else Path(__file__).resolve().parents[1]
    raw,mask=load_observations(root/'data/raw_data.npz')
    result=reconstruct(raw)
    if not np.array_equal(result['masks'][2][None],mask):raise ValueError('public support does not match the selected input convention')
    output=root/'output/reconstruction.npz';output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,reconstruction=result['reconstruction'],correction_factors=result['correction_factors'])
    return result

if __name__=='__main__':
    result=run();print('Reconstructed',int(result['masks'][2].sum()),'supported pixels; factors gamma,delta,epsilon,alpha=',result['correction_factors'][0].tolist())
