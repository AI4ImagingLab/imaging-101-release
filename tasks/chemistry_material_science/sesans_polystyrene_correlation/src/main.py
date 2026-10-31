import json
import numpy as np
from .preprocessing import ROOT,load_observations
from .solvers import reconstruct
def main():
    result,info=reconstruct(load_observations());(ROOT/'output').mkdir(exist_ok=True)
    np.savez(ROOT/'output/reconstruction.npz',**result);print(json.dumps(info));return result,info
if __name__=='__main__':main()
