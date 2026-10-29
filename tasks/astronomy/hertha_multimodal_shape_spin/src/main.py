"""Standard CPU entry point for the complete selected native observation fit."""
from pathlib import Path
import argparse,json,time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

ROOT=Path(__file__).resolve().parents[1]

def run(data_path=None,output_path=None,progress=None):
    """Actively fit all bound data and write the one declared submission NPZ."""
    data_path=ROOT/'data/raw_data.npz' if data_path is None else Path(data_path)
    output_path=ROOT/'output/reconstruction.npz' if output_path is None else Path(output_path)
    output,record=reconstruct(load_observations(data_path),progress=progress)
    output_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output_path,**output)
    return output,record

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'data/raw_data.npz');parser.add_argument('--output',type=Path,default=ROOT/'output/reconstruction.npz');args=parser.parse_args(argv)
    start=time.perf_counter();_,record=run(args.data,args.output,progress=lambda row:print(json.dumps(row),flush=True))
    print(json.dumps(dict(output=str(args.output),iterations=len(record['history']),runtime_seconds=time.perf_counter()-start)),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
