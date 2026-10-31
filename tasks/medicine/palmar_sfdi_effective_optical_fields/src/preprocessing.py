import numpy as np

def demodulate(phase):
    a,b,c=np.asarray(phase,dtype=np.float32)
    dc=(a+b+c)/np.float32(3)
    ac=np.sqrt(np.float32(2)*((a-b)**2+(b-c)**2+(c-a)**2))/np.float32(3)
    return dc,ac

def calibrate(hand,phantom,known_reflectance):
    dh,ah=demodulate(hand[0]);dr,ar=demodulate(phantom[0]);result=np.empty((500,400,5),dtype=np.float32);result[:,:,0]=np.float32(known_reflectance[0])*(dh/dr)
    for i in range(1,5):
        _,ah=demodulate(hand[i]);_,ar=demodulate(phantom[i]);result[:,:,i]=np.float32(known_reflectance[i])*(ah/ar)
    return result

def average_blocks(reflectance,block=10):
    h,w,c=reflectance.shape
    if h%block or w%block:raise ValueError('All observed pixels must belong to complete blocks')
    return np.asfortranarray(reflectance).reshape(h//block,block,w//block,block,c).mean(3).mean(1)

def load_data(path):
    with np.load(path,allow_pickle=False) as a:hand=a['hand_phase'][0].copy();phantom=a['phantom_phase'][0].copy();rcal=a['phantom_reflectance'][0].copy();freq=a['frequency_mm'][0].copy()
    if hand.shape!=(5,3,500,400) or phantom.shape!=hand.shape:raise ValueError('Invalid phase dimensions')
    if not all(np.all(np.isfinite(v)) for v in [hand,phantom,rcal,freq]):raise ValueError('Nonfinite input')
    cal=calibrate(hand,phantom,rcal);return hand,phantom,rcal,freq,cal,average_blocks(cal)
