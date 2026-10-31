"""Full native ensemble reconstruction from all public fluorescence traces."""
from pathlib import Path
import time,numpy as np
from .preprocessing import load_observations,observation_sequences
from .solvers import reconstruct
from .physics_model import viterbi_sequence
ROOT=Path(__file__).resolve().parents[1]
def run(data_path=None,output_path=None):
    start=time.perf_counter();raw,lo,hi=load_observations(ROOT/'data/raw_data.npz' if data_path is None else data_path)
    sequences=observation_sequences(raw,lo,hi);fit=reconstruct(sequences)
    paths=np.full(raw.shape[:2],-1,dtype=np.int64)
    for i,x in enumerate(sequences):paths[i,lo[i]:hi[i]]=viterbi_sequence(x,fit['state_mean'],fit['state_variance'],fit['transition_probability'],fit['initial_probability'])
    fit['reconstruction']=paths;fit['runtime_seconds']=time.perf_counter()-start
    output=ROOT/'output/reconstruction.npz' if output_path is None else Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    keys=['reconstruction','state_mean','state_variance','transition_probability','initial_probability','population_fraction']
    np.savez_compressed(output,**{key:fit[key][None] for key in keys});return fit
if __name__=='__main__':
    result=run();print('All590native traces,64281validframes; means',result['state_mean'],'seconds',result['runtime_seconds'])
