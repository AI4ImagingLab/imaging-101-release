"""Standard whole-field entry point; importing this module selects no backend."""
from pathlib import Path
import argparse,os,json,time

def run(root=None):
 """Reconstruct all40 projections and both100x100 fields; return result/diagnostics."""
 from .preprocessing import load_data,make_model
 from .solvers import reconstruct
 root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
 data=load_data(root/'data/raw_data.npz');model=make_model(data)
 return reconstruct(data['projections'],model)

def main():
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'output/reconstruction.npz');args=parser.parse_args()
 import numpy as np
 start=time.perf_counter();result,diagnostics=run();args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,reconstruction=result[None])
 print(json.dumps(dict(seconds=time.perf_counter()-start,iterations=diagnostics['iterations'],relative_update=diagnostics['relative_update'],shape=list(result[None].shape))))

if __name__=='__main__':main()
