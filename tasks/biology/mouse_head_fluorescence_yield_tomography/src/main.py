"""Standard CPU entry point; importing it does not alter plotting backends."""
from pathlib import Path
import argparse,time,json
import numpy as np
from .preprocessing import load_raw
from .solvers import reconstruct
ROOT=Path(__file__).resolve().parents[1]
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'data/raw_data.npz');p.add_argument('--output',type=Path,default=Path('output/reconstruction.npz'));a=p.parse_args(argv);start=time.perf_counter();result=reconstruct(load_raw(a.input));a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,reconstruction=result['reconstruction'][None].astype(np.float64));print(json.dumps({'runtime_seconds':time.perf_counter()-start,'forward_relative_L2':result['forward_relative_L2'],'objective_evaluations':len(result['objective_history'])}));return 0
if __name__=='__main__':raise SystemExit(main())
