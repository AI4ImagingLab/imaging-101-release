import json
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import fit_sweep

def run(output='output/reconstruction.npz'):
    data=load_data();parameters=[];outputs={};records={}
    for angle,(x,y) in data.items():
        parameter,prediction,record=fit_sweep(x,y,angle)
        parameters.append(parameter);outputs[f'reconstruction_{angle}']=prediction[None];records[angle]=record
    parameters=np.array(parameters)
    outputs.update(magnetic_field=parameters[:,0][None],temperature=parameters[:,1][None],polarization=parameters[:,2][None])
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    np.savez(output,**outputs)
    return outputs,records

if __name__=='__main__':
    outputs,records=run();print(json.dumps(records,indent=2))
