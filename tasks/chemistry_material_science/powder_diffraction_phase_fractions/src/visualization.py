import numpy as np
def composition_rmse(estimate,truth):
    return np.sqrt(np.mean((estimate[:,1:]-truth[:,1:])**2,axis=1))
def observation_rwp(prediction,counts):
    return np.sqrt(np.sum((prediction-counts)**2/counts,axis=1)/np.sum(counts,axis=1))
