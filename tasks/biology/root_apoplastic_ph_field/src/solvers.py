"""Source-preserving direct calibration and dense per-pixel pH inverse."""
import numpy as np
from .preprocessing import valid_pixels,calibration_levels
from .physics_model import normalize_ratio,invert_response


def invert_image(ratio,lower,upper):
    """Return pH float32, input-only valid mask and clipping bool [y,x].

    Float32 matches the original ImageJ FloatProcessor storage. Missing
    source pixels are zero in the declared output, never included in scores.
    """
    mask=valid_pixels(ratio);out=np.zeros(np.shape(ratio),dtype=np.float32)
    clipped=np.zeros(mask.shape,dtype=bool)
    values,saturated=invert_response(normalize_ratio(np.asarray(ratio)[mask],lower,upper))
    out[mask]=values.astype(np.float32);clipped[mask]=saturated
    return out,mask,clipped


def reconstruct(ratios,low_calibration,high_calibration):
    """Return dense float32 [image,y,x], lower/upper levels and saturation rates."""
    lower,upper,_,_=calibration_levels(low_calibration,high_calibration)
    output=[];rates=[]
    for ratio in ratios:
        ph,mask,clipped=invert_image(ratio,lower,upper)
        output.append(ph);rates.append(float(clipped.sum()/mask.sum()))
    return np.stack(output),lower,upper,np.asarray(rates)
