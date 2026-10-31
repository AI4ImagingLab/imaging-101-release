"""Full 128-cubed,500-step joint density/view deep-prior training."""
import time
import os
import numpy as np
import torch
from .network import build_network
from .physics_model import project,objective

SEED=20261026
ITERATIONS=500

def initialize(size=128,seed=SEED,device='cuda'):
    """Return fixed Gaussian input and complete network; native RNG order.

    CUDA input noise is generated before CPU parameter initialization. One
    training-mode warmup forward updates BN buffers as in the author demo.
    """
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    noise=torch.randn(1,3,size,size,size,device=device)
    network=build_network().to(device);network(noise)
    return noise,network

def reconstruct(silhouettes,prior,iterations=ITERATIONS,seed=SEED,device='cuda',view_weight=.1,callback=None,capture_stages=False):
    """Return density[N,N,N],views[V,3],history[steps,2],diagnostics.

    Standard inputN128,full500steps. Density/projections are the final loop
    forward BEFORE the500th update, preserving the native saved result;
    views are final post-update parameters. This sub-step distinction is
    explicitly recorded, rather than silently adding another network pass.
    """
    if iterations<1:raise ValueError('At least one update required')
    started=time.perf_counter();noise,net=initialize(silhouettes.shape[0],seed,device)
    observed=torch.as_tensor(silhouettes,dtype=torch.float32,device=device)
    prior_tensor=torch.as_tensor(prior,dtype=torch.float32,device='cpu')
    views=torch.nn.Parameter(prior_tensor.detach().clone())
    optimizer=torch.optim.Adam(list(net.parameters())+[views],lr=.01)
    history=[];stage={}
    for i in range(iterations):
        optimizer.zero_grad();volume=net(noise)[0,0]
        if capture_stages and i==0:volume.retain_grad()
        projections=project(volume,views)
        loss,data=objective(projections,observed,views,prior_tensor,view_weight)
        if capture_stages and i==0:
            stage.update(output_initial_volume=volume.detach().cpu().numpy().copy(),output_initial_projections=projections.detach().cpu().numpy().copy(),output_first_layer_before=next(net.parameters()).detach().cpu().numpy().copy(),output_initial_loss=np.array([float(loss.detach())]))
        loss.backward()
        if capture_stages and i==0:
            stage.update(output_first_volume_gradient=volume.grad.detach().cpu().numpy().copy(),output_first_view_gradient=views.grad.detach().cpu().numpy().copy(),output_first_layer_gradient=next(net.parameters()).grad.detach().cpu().numpy().copy())
        optimizer.step()
        if capture_stages and i==0:
            stage.update(output_first_layer_after=next(net.parameters()).detach().cpu().numpy().copy(),output_first_views_after=views.detach().cpu().numpy().copy())
        history.append([float(data.detach()),float(loss.detach())])
        if callback is not None:callback(i+1,history[-1])
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    diagnostics={'projections':projections.detach().cpu().numpy(),'runtime_seconds':time.perf_counter()-started,'stages':stage}
    return volume.detach().cpu().numpy(),views.detach().cpu().numpy(),np.array(history),diagnostics
