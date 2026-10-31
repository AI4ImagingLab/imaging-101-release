"""Standard task entry point; importing it does not alter plotting backends."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import infer_distortion
from .physics_model import rectify
def run(data_path=None,output_path='output/reconstruction.npz'):
    data=load_data(data_path);a=infer_distortion(data['line_points'],data['line_ids'])
    result={'reconstruction':rectify(data['image'],a)[None],'distortion':a[None]}
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**result)
    return result
if __name__=='__main__':run()
