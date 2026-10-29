from pathlib import Path
import numpy as np
from .preprocessing import load_data,normalize
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    image,offsets,mask=load_data(data_path)
    spectra,offsets,_=normalize(image,offsets)
    parameters,field=reconstruct(spectra,offsets,mask)
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,amplitude=parameters[None,...,0],width=parameters[None,...,1],center=parameters[None,...,2],field_shift=field[None])
    return parameters,field

if __name__=='__main__':run()
