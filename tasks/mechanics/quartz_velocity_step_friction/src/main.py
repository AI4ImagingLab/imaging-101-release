from pathlib import Path
import time,json,numpy as np
from .preprocessing import prepare
from .solvers import reconstruct

def main():
    start=time.perf_counter();data=prepare();result,info=reconstruct(data)
    Path('output').mkdir(exist_ok=True);np.savez_compressed('output/reconstruction.npz',**result)
    print(json.dumps({'runtime_s':time.perf_counter()-start,'parameters':result['reconstruction'].tolist(),'nfev':info['nfev'],'objective':info['objective'],'success':info['success']}))
    return result,info

if __name__=='__main__':main()
