"""Published Poisson ordered-subsets EM, explicit preconditioned update."""
import torch
from .physics_model import DELTA


@torch.no_grad()
def reconstruct(acquisition, photopeak, scatter, n_iterations=4, capture=False):
    """Return full activity tensor and optional 32 subset states; no support mask."""
    g = torch.as_tensor(photopeak,dtype=torch.float32,device=acquisition.device)
    s = torch.as_tensor(scatter,dtype=torch.float32,device=acquisition.device)
    image = torch.ones(acquisition.shape,device=acquisition.device)
    norms = [acquisition.backward(torch.ones_like(g[i::acquisition.n_subsets]),i)
             for i in range(acquisition.n_subsets)]
    states = []
    for iteration in range(n_iterations):
        for subset,norm in enumerate(norms):
            indices = slice(subset,None,acquisition.n_subsets)
            prediction = acquisition.forward(image,subset)+s[indices]
            ratio = g[indices]/(prediction+DELTA)
            gradient = acquisition.backward(ratio,subset)-norm
            preconditioner = image*(1/(norm+DELTA))
            image += preconditioner*gradient
            image[image<=0] = 0
            if capture: states.append(image.detach().cpu().numpy().copy())
    return image,states
