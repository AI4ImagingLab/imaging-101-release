from pathlib import Path
import numpy as np

def load_data(path=Path('data/raw_data.npz')):
    with np.load(path,allow_pickle=False) as data:return data['velocity'][0],data['transmission'][0]
