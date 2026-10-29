"""Standard standalone task entry point."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations, ROOT
from .solvers import reconstruct

def save_diagnostics(result, path):
    """Optional maintainer numeric capture; writes only, never loads a cache."""
    arrays = {k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    arrays.update({f'snapshot_{k}':v for k,v in result['snapshots'].items()})
    arrays.update(iterations=np.array(result['iterations']),
                  runtime_seconds=np.array(result['runtime_seconds']),
                  negative_determinant_count=np.array(result['negative_determinant_count']))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)

def main(diagnostics=None):
    data = load_observations()
    result = reconstruct(data['x'], data['y'], data['fluence'])
    output = ROOT / 'output'; output.mkdir(exist_ok=True)
    np.savez_compressed(output / 'reconstruction.npz', reconstruction=result['reconstruction'][None])
    if diagnostics is not None:
        save_diagnostics(result, diagnostics)
    print(f"{result['iterations']} updates; {result['stopping']}; {result['runtime_seconds']:.3f} seconds")
    return result

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostics',type=Path)
    main(parser.parse_args().diagnostics)
