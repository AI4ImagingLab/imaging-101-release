import numpy as np
NOMINAL_PIXELS_PER_METRE=160/.007
EFFECTIVE_PIXELS_PER_METRE=NOMINAL_PIXELS_PER_METRE
FSIGMA=140.
RADIUS=72.92296540262036
CENTER=(93.26971254206686,90.04002272424479)
def load_data(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k][0] for k in f.files}
def prepare(rgb):
    return np.clip((rgb[:,:,1].astype(np.float64)-.5*rgb[:,:,0])/(255*.65),0.,1.)
def disk_mask(shape=(181,181),center=CENTER,radius=RADIUS):
    y,x=np.ogrid[:shape[0],:shape[1]]
    return (x-center[1])**2+(y-center[0])**2<=radius**2
