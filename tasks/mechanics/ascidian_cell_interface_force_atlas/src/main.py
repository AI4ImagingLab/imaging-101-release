"""Standard entry point: python -m src.main."""
from pathlib import Path
import argparse,time,json
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def main(argv=None):
    parser=argparse.ArgumentParser(description='Infer interface tensions and cell pressures from observed 3D membrane geometry')
    parser.add_argument('--data',type=Path,default=Path('data/raw_data.npz'))
    parser.add_argument('--output',type=Path,default=Path('output/reconstruction.npz'))
    args=parser.parse_args(argv)
    start=time.perf_counter();r=reconstruct(load_observations(args.data))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,tensions=r['tensions'][None],pressures=r['pressures'][None])
    info={'runtime_seconds':time.perf_counter()-start,'interfaces':len(r['tensions']),'regions':len(r['pressures']),
          'malformed_triple_edges':len(r['geometry']['malformed_edges']),'output':str(args.output)}
    print(json.dumps(info,indent=2))
    return info

if __name__=='__main__':main()
