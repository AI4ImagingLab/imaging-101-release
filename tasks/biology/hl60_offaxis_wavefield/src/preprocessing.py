from pathlib import Path
import numpy as np
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as f:return {k:f[k][0].copy() for k in f.files}
def padded_spectrum(image,padding=2):
    image=image.astype(np.float64);image-=image.mean()
    size=int(2**np.ceil(np.log(padding*max(image.shape))/np.log(2)))
    padded=np.zeros((size,size));padded[:image.shape[0],:image.shape[1]]=image
    return np.fft.fftshift(np.fft.fft2(padded))
