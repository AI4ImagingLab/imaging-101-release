"""Standard active reconstruction; no upstream imports or cached results."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data,T2_MS,LAMBDA
from .physics_model import build_dictionary,predict
from .solvers import solve_distribution

def reconstruct(data_path,ridge=LAMBDA):
    d=load_data(data_path);A=build_dictionary(d['cluster_centroids'],T2_MS)
    x=solve_distribution(A,d['y'],d['labels'],ridge);curve=predict(A,x,d['labels']);total=x.sum(axis=1)
    mw=np.divide(x[:,(T2_MS>15)&(T2_MS<=40)].sum(axis=1),total,out=np.zeros_like(total),where=total>0)
    mass=np.zeros((120,120,100));density=np.zeros((120,120));fraction=np.zeros((120,120));pred=np.zeros((120,120,32))
    mass[d['mask']]=x;density[d['mask']]=total;fraction[d['mask']]=mw;pred[d['mask']]=curve
    return dict(reconstruction=mass[None],myelin_fraction=fraction[None],total_density=density[None],predicted_echoes=pred[None]),dict(data=d,dictionary=A,coefficients=x)

def main():
    root=Path(__file__).resolve().parents[1];start=time.perf_counter();result,_=reconstruct(root/'data/raw_data.npz')
    (root/'output').mkdir(exist_ok=True);np.savez_compressed(root/'output/reconstruction.npz',**result)
    print(f'Fresh complete-plane reconstruction: {time.perf_counter()-start:.6f}s')

if __name__=='__main__':main()
