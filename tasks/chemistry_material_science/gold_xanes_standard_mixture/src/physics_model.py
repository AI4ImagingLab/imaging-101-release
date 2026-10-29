"""Local spectral mixing equations; no domain-library calls."""
import numpy as np
def forward(weights,standards):
    weights=np.asarray(weights);standards=np.asarray(standards)
    return (weights[0]*standards[0]+weights[1]*standards[1]+weights[2]*standards[2]+
            weights[3]*standards[3]+weights[4]*standards[4]+weights[5]*standards[5])
def bounded_weights(internal,model):
    # lmfit finite [0,1] sine transform; final expression is clipped natively.
    free=(np.sin(internal)+1.)/2.
    w=np.zeros(6);w[list(model[:-1])]=free
    remainder=1.
    for value in free:remainder-=value
    w[model[-1]]=np.clip(remainder,0.,1.)
    return w
def residual(internal,model,observation,standards,scale=.001):
    return (observation-forward(bounded_weights(internal,model),standards))/scale
def affine_jacobian(model,standards,scale=.001):
    # Derivative with respect to physical free weights in the interior simplex.
    return -(standards[list(model[:-1])]-standards[model[-1]]).T/scale
