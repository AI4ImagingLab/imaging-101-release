import numpy as np
from scipy.sparse import csr_matrix

def antialiased_triangle_matrix(input_size,output_size,center_mode=0):
    scale=(input_size-1)/(output_size-1)
    support=max(1.,scale)
    rows=[];cols=[];values=[]
    for i in range(output_size):
        center=scale*(i+.5) if center_mode==0 else scale*i+.5
        start=max(int(center-support+.5),0)
        stop=min(int(center+support+.5),input_size)
        x=np.arange(start,stop)
        v=np.maximum(0,1-np.abs((x-center+.5)/support))
        v=v/v.sum()
        rows.extend([i]*len(x));cols.extend(x);values.extend(v.astype(np.float32))
    return csr_matrix((np.asarray(values,dtype=np.float32),(rows,cols)),shape=(output_size,input_size))

def resize(frame,shape,center_mode=0):
    y=antialiased_triangle_matrix(frame.shape[0],shape[0],center_mode)
    x=antialiased_triangle_matrix(frame.shape[1],shape[1],center_mode)
    return (x@(y@frame.astype(np.float32)).T).T



def load_observations(path):
    with np.load(path,allow_pickle=False) as z:
        images=z["images"]
    if images.shape!=(1,100,1040,1388) or images.dtype!=np.uint8:
        raise ValueError("Expected original batch of100uint8frames")
    return images[0]

def resize_stack(images,shape):
    wy=antialiased_triangle_matrix(images.shape[-2],shape[0])
    wx=antialiased_triangle_matrix(images.shape[-1],shape[1])
    out=np.empty((len(images),*shape),dtype=np.float32)
    for i,frame in enumerate(images):
        out[i]=(wx@(wy@frame.astype(np.float32)).T).T
    return out

def upsample_field(field,shape=(1040,1388)):
    wy=antialiased_triangle_matrix(field.shape[0],shape[0],center_mode=1)
    wx=antialiased_triangle_matrix(field.shape[1],shape[1],center_mode=1)
    return (wx@(wy@field).T).T
