"""Standard independent command-line reconstruction; no plotting side effects."""
from pathlib import Path
import argparse,os,json

def run(data_path=None,output_path=None,initial=None):
    """Actively fit observations, write one submission, return (output,record)."""
    import numpy as np
    from .preprocessing import load_observations
    from .solvers import reconstruct
    root=Path(__file__).resolve().parents[1]
    data=load_observations(root/'data/raw_data.npz' if data_path is None else data_path)
    output,record=reconstruct(data,initial)
    destination=root/'output/reconstruction.npz' if output_path is None else Path(output_path)
    destination.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(destination,**output)
    return output,record

def main(argv=None):
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
    parser=argparse.ArgumentParser();parser.add_argument('--data');parser.add_argument('--output');args=parser.parse_args(argv)
    output,record=run(args.data,args.output)
    print(json.dumps(dict(parameters=output['parameters'].tolist(),nfev=record['nfev'],
        seconds=record['seconds'],success=record['success'])))
    return 0

if __name__=='__main__':raise SystemExit(main())
