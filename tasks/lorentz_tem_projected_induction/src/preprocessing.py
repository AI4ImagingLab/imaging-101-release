"""BSD-3-Clause adaptation of PyLorentz TIE._select_images/_scale_stack.
Copyright (c) PyLorentz contributors; see LICENSE and private validation.
"""
import numpy as np
DEFOCUS_NM=np.array([-928972.,-398131.,-199065.,0.,199065.,398131.,928972.])
PIXEL_NM=5.1066
VOLTAGE_V=200000.
def load_observations(path):
    """Load two batch-first (1,7,512,512) float32 count stacks."""
    with np.load(path,allow_pickle=False) as z:
        a=z['unflipped'];b=z['flipped']
    for x in [a,b]:
        if x.shape!=(1,7,512,512) or x.dtype!=np.float32 or not np.isfinite(x).all() or x.min()<0:
            raise ValueError('expected finite nonnegative float32 (1,7,512,512) observations')
    return a[0],b[0]
def prepare(unflipped,flipped,index=0):
    """Return selected equal-flux normalized images, common I0, B/E differences.

    Preserve source float32 reduction/multiplication, +1e-9 floor, and
    zero-mean difference convention. Index0 selects the nearest focus pair.
    """
    if not isinstance(index,int) or isinstance(index,bool) or index not in [0,1,2]:raise ValueError('index must be 0,1,2')
    if unflipped.shape!=flipped.shape or unflipped.shape[0]!=7:raise ValueError('matching seven-image series required')
    m=3;k=index+1
    infocus=(unflipped[m]+flipped[m])/2
    stack=np.stack([unflipped[m-k],flipped[m-k],infocus,unflipped[m+k],flipped[m+k]])
    stack=stack.copy();totals=np.sum(stack,axis=(1,2))
    if (totals<=0).any():raise ValueError('zero photon flux')
    stack*= (np.max(totals)/totals)[:,None,None]
    stack=stack/np.max(stack)+1e-9
    db=.5*((stack[3]-stack[0])-(stack[4]-stack[1]))
    de=.5*((stack[3]-stack[0])+(stack[4]-stack[1]))
    # Source mask is float64 all ones, while in-place subtraction stays float32.
    for d in [db,de]:d-=d.sum()/np.ones(d.shape).sum()
    return stack,stack[2],db,de
