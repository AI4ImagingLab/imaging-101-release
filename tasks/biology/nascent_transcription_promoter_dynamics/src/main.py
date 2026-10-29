"""Full corrected published branch, all real observations, no cached inference."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import PromoterModel
def reconstruct(path=None):
    values,lengths,processed=load_data(path)
    np.random.seed(451134116)
    model=PromoterModel(2,19,30,20,1.5,2**19,processed)
    initial=model.initialise_parameters()
    priors=model.EM_fixed(initial,5,len(lengths),32,.01,451134116)
    final=model.EM_with_priors(priors,100,len(lengths),32,.001,451134116)
    posterior=model.get_promoter_traces(final,1,len(lengths),32,.001,451134116)
    result=np.zeros_like(values)
    for i,n in enumerate(lengths):result[i,:n]=np.exp(posterior[i])[1]
    output=dict(posterior=result[None],transition=final['A'][None],emission=final['v'].reshape(1,2),noise=np.array([[final['noise']]]),initial=final['pi0'].reshape(1,2))
    return output
def main():
    result=reconstruct();p=Path('output');p.mkdir(exist_ok=True);np.savez_compressed(p/'reconstruction.npz',**result)
if __name__=='__main__':main()
