import numpy as np
import torch
from .physics_model import losses,tensors,TV_WEIGHT

def reconstruct(raw,corrected,initial,weight=TV_WEIGHT,max_epochs=200000,patience=100,stop_loss=1e-5):
    torch.set_num_threads(1)
    inputs=tensors(raw,corrected);energy=torch.nn.Parameter(torch.tensor(np.asarray(initial),dtype=torch.float64));opt=torch.optim.SGD([energy],lr=1e4,momentum=.5,nesterov=True)
    history=[];previous=float('inf');stale=0
    for epoch in range(max_epochs):
        opt.zero_grad();mse,tv,pred=losses(energy,*inputs,weight=weight);loss=mse+tv;loss.backward();opt.step();history.append([mse.item(),tv.item()]);value=loss.item();diff=previous-value
        stale=stale+1 if diff<stop_loss else 0;previous=value
        if stale>patience:break
    with torch.no_grad():mse,tv,pred=losses(energy,*inputs,weight=weight)
    return energy.detach().numpy().copy(),pred.numpy().copy(),np.array(history),float(mse+tv)
