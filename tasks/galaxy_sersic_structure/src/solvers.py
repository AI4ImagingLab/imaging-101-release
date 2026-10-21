"""Local constrained Gaussian MAP with the published three-round Adam schedule.
Adapted mathematical method from PySersic, MIT LICENSE.
"""
import copy,numpy as np,torch
from .physics_model import PixelModel,objective
LOW=np.array([0.,.5,.65,0.,0.],dtype=np.float32)
HIGH=np.array([300.,30.,8.,.9,np.pi],dtype=np.float32)
# Original init_to_median/SVI.init pure-prior draws, seeds20261020/21/22.
# Captured before optimization; no observation or fitted answer enters these.
INITIAL_LATENT=np.array([
[-.8844311833381653,.10488099604845047,-.0666516125202179,-.5177764296531677,.821110188961029,.861896276473999,.7182143330574036,.4267004728317261],
[-.8412160277366638,-.41380053758621216,-.3878434896469116,.22249045968055725,-.6993310451507568,-.5866261124610901,.486686646938324,.363640695810318],
[-.8512426614761353,-.06809680163860321,.2090480774641037,-.01715460792183876,-.022554848343133926,-.06700687110424042,-.055138226598501205,-.47660836577415466]],dtype=np.float32)
def physical_parameters(latent):
    bounded=torch.as_tensor(LOW)+torch.as_tensor(HIGH-LOW)*torch.sigmoid(latent[2:7])
    return torch.cat([60+5*latent[:2],bounded,.01*latent[7:]])
def fit_band(image,inverse_variance,psf,mask,max_train=20000,patience=250,initial_latent=None):
    """Return(physical8-vector,intrinsic120x120,observed120x120,history).
    Gaussian position/sky priors and bounded uniform structural priors.
    Default initialization is the captured original g-band pure-prior seed;
    reconstruct supplies each band's original optimizer-unconstrained draw.
    """
    model=PixelModel(psf);im=torch.tensor(image,dtype=torch.float32);rms=torch.tensor((1/np.sqrt(inverse_variance)).astype(np.float32));valid=torch.tensor(~mask.astype(bool))
    initial=torch.tensor(INITIAL_LATENT[0]if initial_latent is None else initial_latent,dtype=torch.float32)
    z=torch.nn.Parameter(initial);optimizer=torch.optim.Adam([z],lr=.05,betas=(.9,.999),eps=1e-8);history=[]
    # Original train_numpyro_svi_early_stop performs one warm stable_update.
    optimizer.zero_grad();loss=objective(physical_parameters(z),model,im,rms,valid)
    if not torch.isfinite(loss):raise ValueError('Nonfinite initial objective')
    loss.backward();optimizer.step();best_loss=float(loss.detach())
    best_z=z.detach().clone();best_optimizer=copy.deepcopy(optimizer.state_dict())
    for round_index in range(3):
        if best_z is not None:
            with torch.no_grad():z.copy_(best_z)
            optimizer.load_state_dict(copy.deepcopy(best_optimizer))
        for group in optimizer.param_groups:group['lr']=.05*.1**round_index
        if round_index>0:best_loss=float('inf')
        wait=0
        for step in range(max_train):
            optimizer.zero_grad();loss=objective(physical_parameters(z),model,im,rms,valid)
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            loss.backward();optimizer.step();value=float(loss.detach())
            if value<best_loss:
                best_loss=value;best_z=z.detach().clone();best_optimizer=copy.deepcopy(optimizer.state_dict());wait=0
            elif wait>=patience:break
            else:wait+=1
            history.append(value)
    with torch.no_grad():
        z.copy_(best_z);params=physical_parameters(z);intrinsic=model.intrinsic(params);observed=model.observed(params)
    return params.numpy(),intrinsic.numpy(),observed.numpy(),np.asarray(history)
def reconstruct(data,return_diagnostics=False):
    params=[];intrinsic=[];observed=[];history=[]
    for band,(image,weight,psf)in enumerate(zip(data['images'],data['inverse_variance'],data['psfs'])):
        p,i,o,h=fit_band(image,weight,psf,data['mask'][0],initial_latent=INITIAL_LATENT[band]);params.append(p);intrinsic.append(i);observed.append(o);history.append(h)
    result=dict(parameters=np.array(params)[None],reconstruction=np.array(intrinsic)[None])
    return(result,dict(observed=np.array(observed),history=history))if return_diagnostics else result
