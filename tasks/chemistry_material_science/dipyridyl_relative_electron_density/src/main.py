"""Standard task entry point; reconstructs only public measured observations."""
from pathlib import Path
import argparse,json
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

ROOT=Path(__file__).resolve().parents[1]

def main(diagnostics=None):
    data=load_observations(ROOT/'data/raw_data.npz');result=reconstruct(data)
    (ROOT/'output').mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'output/reconstruction.npz',reconstruction=result['reconstruction'][None])
    if diagnostics:
        path=Path(diagnostics);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**{k:result[k] for k in ['reconstruction','history','origin_shift','origin_score','origin_terms','runtime_seconds']},**{'snapshot_'+str(k):v for k,v in result['snapshots'].items()})
    print(json.dumps({'iterations':result['iterations'],'runtime_seconds':result['runtime_seconds'],'shape':list(result['reconstruction'].shape),'last_Rcf_percent':float(result['history'][-1,1])}))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--diagnostics',type=Path);args=parser.parse_args();main(args.diagnostics)
