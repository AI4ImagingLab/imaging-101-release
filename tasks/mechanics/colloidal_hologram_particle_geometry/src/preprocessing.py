from pathlib import Path
import numpy as np
ROIS=((809,187,228,228),(222,635,200,200))
def load(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as f: image=f['image'][0]; background=f['background'][0]
    return image,background,(image.astype(float)-13)/(background.astype(float)-13)
def regions(normalized):
    for x,y,w,h in ROIS:
        crop=normalized[y:y+h,x:x+w]; yy,xx=np.indices(crop.shape)
        coords=np.array([(xx+x).ravel(),(yy+y).ravel(),np.zeros(xx.size)])
        valid=(crop!=crop.max()).ravel()
        yield crop,coords,valid
