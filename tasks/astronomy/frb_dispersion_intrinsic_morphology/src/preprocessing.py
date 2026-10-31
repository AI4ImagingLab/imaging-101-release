import numpy as np

def load(path="data/raw_data.npz"):
 with np.load(path,allow_pickle=False) as f:return {k:f[k][0] for k in f.files}

def weights(data):
 w=np.zeros(len(data['frequencies']));good=data['good_channels'];w[good]=1/data['noise_rms'][good];return w
