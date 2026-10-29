"""Known-probe surface inversion and explicitly labeled blind comparison."""
import numpy as np
from .physics_model import erosion

def reconstruct(measurements, probe, nonnegative=True):
    """Return float64 surfaces [frame,y,x], in A; optionally impose substrate 0.

    The shipped method is finite-grid morphological erosion followed by a
    nonnegative substrate projection, not an iterative or least-squares method.
    """
    surface=erosion(measurements,probe)
    return np.maximum(surface,0.0) if nonnegative else surface

def blind_comparison(measurements, epochs=200, tip_size=10, lr=0.1, weight_decay=0.0011):
    """Fresh published differentiable-BTR comparison, not the shipped solver.

    Reproduce the modern author's notebook: float32 input, zero probe, AdamW,
    mean opening error, nonpositive clamp and high-tip center-of-mass recenter.
    Returns float32 (surfaces[frame,y,x], tip[K,K], losses[epochs]).
    Its same-offset erosion convention deliberately preserves the upstream
    comparison; use physics_model.erosion for the general order adjoint.
    """
    import torch
    import torch.nn.functional as F
    x=torch.as_tensor(np.asarray(measurements),dtype=torch.float32)
    p=torch.zeros((tip_size,tip_size),dtype=x.dtype,requires_grad=True)
    opt=torch.optim.AdamW([p],lr=lr,weight_decay=weight_decay)
    lo=(tip_size-1)//2;hi=tip_size-1-lo
    def morph(a,t,erode=False):
        v=F.pad(a[None,None],(lo,hi,lo,hi),value=float("inf") if erode else -float("inf"))
        v=F.unfold(v,tip_size).reshape(tip_size*tip_size,-1)
        v=t.reshape(-1,1)-v if erode else t.reshape(-1,1)+v
        z=v.max(dim=0).values
        return (-z if erode else z).reshape(a.shape)
    def recenter(t):
        floor=t.min();weight=t-floor
        weight[weight<1e-8]=0.0
        if torch.all(weight<1e-10):weight.fill_(1.0)
        ix=torch.cumsum(torch.ones_like(t),dim=0)-1.0
        iy=torch.cumsum(torch.ones_like(t),dim=1)-1.0
        cx,cy=round(float(torch.sum(ix*weight/torch.sum(weight)))),round(float(torch.sum(iy*weight/torch.sum(weight))))
        center=round((tip_size-1)/2)
        ax,bx=max(-center,-cx),min(tip_size-center,tip_size-cx)
        ay,by=max(-center,-cy),min(tip_size-center,tip_size-cy)
        z=torch.full_like(t,float(floor));z[center+ax:center+bx,center+ay:center+by]=t[cx+ax:cx+bx,cy+ay:cy+by]
        return z
    losses=[]
    for _ in range(epochs):
        total=0.0
        for frame in x:
            opt.zero_grad();opened=morph(morph(frame,p,True),p)
            loss=((opened-frame)**2).mean();loss.backward();opt.step()
            with torch.no_grad():p.copy_(recenter(torch.clamp(p,max=0.0)))
            total+=float(((morph(morph(frame,p,True),p)-frame)**2).mean().detach())
        losses.append(total)
    return torch.stack([morph(frame,p,True) for frame in x]).detach().numpy(),p.detach().numpy(),np.asarray(losses)
