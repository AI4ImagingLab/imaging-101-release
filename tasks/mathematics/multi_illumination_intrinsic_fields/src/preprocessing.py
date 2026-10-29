"""Input validation; values preserve the original 16-bit PNG /255 convention."""
from pathlib import Path
import numpy as np
def load_observations(path: str | Path) -> dict:
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'reference_image','illumination_stack','object_mask'}:raise ValueError('Unexpected public array keys')
        image=z['reference_image'];multiple=z['illumination_stack'];mask=z['object_mask']
    if image.dtype!=np.float64 or multiple.dtype!=np.float64 or mask.dtype!=np.bool_:raise ValueError('Invalid observation dtype')
    if image.shape!=(1,600,534) or multiple.shape!=(1,10,600,534) or mask.shape!=(1,600,534):raise ValueError('Invalid observation shape')
    if not np.isfinite(image).all() or not np.isfinite(multiple).all() or np.any(image<0) or np.any(multiple<0):raise ValueError('Nonfinite or negative observation')
    return {'image':image[0],'multiple':np.moveaxis(multiple[0],0,-1),'mask':mask[0]}
