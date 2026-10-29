import numpy as np,pytest
from src.solvers import stretch_posterior

class LinearGaussianObservation:
    def forward(self,parameters,masked=True):return np.array(parameters,copy=True)

@pytest.mark.parametrize('seed',[17,317])
def test_stretch_sampler_gaussian_mean_variance(seed):
    mean=np.array([5.7,4.4,-.1,3.8]);sigma=np.array([.08,.1,.05,.1])
    likelihood=dict(parameters=mean,walker_width=sigma,errors=sigma,masked_flux=mean)
    chain,logp,acceptance=stretch_posterior(LinearGaussianObservation(),likelihood,burn=100,production=500,seed=seed)
    samples=chain.reshape(-1,4)
    assert chain.shape==(16,500,4) and np.isfinite(chain).all() and np.isfinite(logp).all()
    assert np.all(abs(samples.mean(0)-mean)<.2*sigma)
    ratio=np.var(samples,axis=0)/sigma**2;assert np.all((ratio>.75)&(ratio<1.25))
    assert .3<acceptance.mean()<.8
