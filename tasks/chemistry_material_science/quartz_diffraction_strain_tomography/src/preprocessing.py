"""Load existing author-simulated diffraction observations without alteration."""
import numpy as np
SHAPE=(106,96)
COMPONENTS=('xx','yy','zz','xy','xz','yz')
def load_data(path):
    shapes={'strain_measurements':(1,500,192),'diffraction_vectors':(1,3,500,192),'noise_std':(1,500,192),'angles':(1,500),'support':(1,*SHAPE)}
    with np.load(path,allow_pickle=False) as z:
        result={}
        for key,shape in shapes.items():
            a=z[key]
            if a.shape!=shape or a.dtype!=(np.bool_ if key=='support' else np.float64):raise ValueError('invalid '+key)
            if not np.isfinite(a).all():raise ValueError('nonfinite '+key)
            result[key]=a[0].copy()
    if np.any(result['noise_std']<0):raise ValueError('negative noise standard deviation')
    return result

def weights_and_directions(data):
    sigma=data['noise_std'];w=np.divide(1,sigma,out=np.zeros_like(sigma),where=sigma!=0);w/=np.max(w)
    g=data['diffraction_vectors'];k=g/(np.linalg.norm(g,axis=0)+1e-12)
    s=np.array([k[0]**2,k[1]**2,k[2]**2,2*k[0]*k[1],2*k[0]*k[2],2*k[1]*k[2]])
    return w,s
