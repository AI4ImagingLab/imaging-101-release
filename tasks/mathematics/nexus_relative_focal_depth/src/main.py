"""Fresh full native reconstruction; standard entry point python -m src.main."""
from pathlib import Path
import argparse,time,json
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct


def save_stage_measurements(details,path):
    """Save identified fresh numerical measurements, never a resume cache."""
    points=np.array([[0,0],[0,1919],[1079,0],[1079,1919],[500,600],[625,1240],[810,1650],[250,400],[900,950]],np.int64)
    arrays={'points':points,'initial_focal_coordinate':details['initial'][None,points[:,0],points[:,1]]}
    for key,name in [('coefficients','positive_focus_coefficients'),('derivative','negative_focus_derivative'),('sharpness','raw_sharpness')]:
        arrays[name]=details[key][:,points[:,0],points[:,1]]
    for n,state in details['steps'].items():
        for key,array in state.items():arrays[f'step{n}_{key}']=array[points[:,0],points[:,1]]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics',type=Path,default=None)
    args=parser.parse_args();start=time.perf_counter()
    root=Path(__file__).resolve().parents[1]
    result,details=reconstruct(load_observations())
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=result[None])
    if args.diagnostics is not None:save_stage_measurements(details,args.diagnostics)
    print(json.dumps({'iterations':400,'runtime_seconds':time.perf_counter()-start,
        'objective':details['objective'],'shape':list(result[None].shape)},allow_nan=False))


if __name__=='__main__':main()
