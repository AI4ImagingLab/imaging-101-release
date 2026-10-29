"""Independent equation-based implementation of the selected dual-prior model."""
import os
import time
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .physics_model import blur_image,loss_for_step


def conv(cin, cout, size=3, stride=1):
    """Create a biased convolution with reflection rather than zero padding."""
    return nn.Sequential(nn.ReflectionPad2d(size//2),nn.Conv2d(cin,cout,size,stride))


def norm_act(channels):
    """Training-mode batch normalization and in-place slope-0.2 activation."""
    return nn.Sequential(nn.BatchNorm2d(channels),nn.LeakyReLU(.2,inplace=True))


class SpatialInteraction(nn.Module):
    """Unnormalized dot-product nonlocal residual with pooled keys/values."""
    def __init__(self):
        super().__init__()
        self.values=nn.Conv2d(128,64,1)
        self.project=nn.Sequential(nn.Conv2d(64,128,1),nn.BatchNorm2d(128))
        nn.init.zeros_(self.project[1].weight);nn.init.zeros_(self.project[1].bias)
        self.queries=nn.Conv2d(128,64,1)
        self.keys=nn.Conv2d(128,64,1)

    def forward(self, x):
        batch=x.shape[0]
        v=F.max_pool2d(self.values(x),2).view(batch,64,-1).permute(0,2,1)
        q=self.queries(x).view(batch,64,-1).permute(0,2,1)
        k=F.max_pool2d(self.keys(x),2).view(batch,64,-1)
        mixed=((q@k)/k.shape[-1])@v
        mixed=mixed.permute(0,2,1).contiguous().view(batch,64,*x.shape[-2:])
        return self.project(mixed)+x


class Scale(nn.Module):
    """One scale of the five-level image prior; two-stage encode/decode API."""
    def __init__(self, cin, index):
        super().__init__()
        # Allocation follows the published implementation's top-down RNG order.
        self.fusion_norm=nn.BatchNorm2d(144)
        self.side=nn.Sequential(conv(cin,16,1),*norm_act(16))
        self.down=nn.Sequential(conv(cin,128,3,2),*norm_act(128))
        self.context=SpatialInteraction() if index>1 else nn.Identity()
        self.refine=nn.Sequential(conv(128,128),*norm_act(128))
        self.up=nn.Sequential(conv(144,128),*norm_act(128),conv(128,128,1),*norm_act(128))

    def encode(self,x):
        return self.side(x),self.refine(self.context(self.down(x)))

    def decode(self,side,deeper):
        z=F.interpolate(deeper,scale_factor=2,mode='bilinear',align_corners=False)
        h=min(z.shape[-2],side.shape[-2]);w=min(z.shape[-1],side.shape[-1])
        def centered(a):
            r=(a.shape[-2]-h)//2;c=(a.shape[-1]-w)//2
            return a[:,:,r:r+h,c:c+w]
        return self.up(self.fusion_norm(torch.cat((centered(side),centered(z)),dim=1)))


class ImagePrior(nn.Module):
    """Five 128-channel scales, 16-channel skips, and three nonlocal blocks."""
    def __init__(self):
        super().__init__()
        self.scales=nn.ModuleList([Scale(8 if j==0 else 128,j) for j in range(5)])
        self.readout=conv(128,1,1)

    def forward(self,z):
        side=[]
        for layer in self.scales:
            a,z=layer.encode(z);side.append(a)
        for layer,a in zip(reversed(self.scales),reversed(side)):
            z=layer.decode(a,z)
        return torch.sigmoid(self.readout(z))

    def optimization_parameters(self):
        """Yield parameters in acquisition-release module traversal order."""
        for layer in self.scales:
            for branch in (layer.side,layer.down,layer.context,layer.refine):
                yield from branch.parameters()
        for layer in reversed(self.scales):
            yield from layer.fusion_norm.parameters()
            yield from layer.up.parameters()
        yield from self.readout.parameters()


class KernelPrior(nn.Module):
    """A 200-to-1000-to-size-squared generator with simplex-valued output."""
    def __init__(self,size=51):
        super().__init__()
        self.hidden=nn.Linear(200,1000);self.readout=nn.Linear(1000,size*size)
        self.size=size

    def forward(self,z):
        logits=self.readout(F.relu6(self.hidden(z)))
        return torch.softmax(logits,dim=-1).reshape(1,1,self.size,self.size)


@dataclass
class ReconstructionResult:
    reconstruction: np.ndarray
    kernel: np.ndarray
    history: np.ndarray
    runtime_seconds: float


def configure_backend(seed=42):
    """Set deterministic algorithm conventions; modifies Torch global flags."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def initialize_priors(shape, device='cuda', seed=42):
    """Return (image network, kernel network, image noise, kernel noise)."""
    configure_backend(seed)
    h,w=shape
    z=torch.empty(1,8,h+50,w+50).uniform_(0,.1).to(device)
    image=ImagePrior().to(device)
    zk=torch.empty(1,200,1,1).uniform_(0,.1).to(device).squeeze()
    kernel=KernelPrior().to(device)
    return image,kernel,z,zk


def learning_rates(step):
    """Closed-form epoch scheduler, preserving updates at 1600/1900/2200."""
    exponent=sum(step>=milestone for milestone in (1600,1900,2200))
    return .01*.5**exponent,.0001*.5**exponent


def reconstruct(observation, iterations=2500, seed=42, device='cuda', progress=True):
    """Fit both priors to a 2-D float32 observation; return pre-last-update image.

    The standard benchmark requires all 2500 iterations. Shorter paths exist only
    for numerical stage tests. Each returned image is evaluated, never loaded.
    """
    if observation.ndim != 2 or observation.dtype != np.float32:
        raise ValueError('observation must be a 2-D float32 array')
    if not np.isfinite(observation).all() or observation.min()<0 or observation.max()>1:
        raise ValueError('observation must be finite and in [0,1]')
    if iterations < 1 or int(iterations) != iterations:
        raise ValueError('iterations must be a positive integer')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the validated full benchmark')
    start=time.perf_counter()
    image,kn,z,zk=initialize_priors(observation.shape,device,seed)
    target=torch.from_numpy(observation[None,None]).to(device)
    optimizer=torch.optim.Adam([{'params':image.optimization_parameters()},{'params':kn.parameters(),'lr':.0001}],lr=.01)
    trace=[]
    iterator=range(iterations)
    if progress:
        from tqdm import tqdm
        iterator=tqdm(iterator,desc='Joint blind reconstruction')
    h,w=observation.shape
    for step in iterator:
        perturbed=z+.001*torch.zeros_like(z).normal_()
        rates=learning_rates(step)
        for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate
        optimizer.zero_grad()
        latent=image(perturbed);kernel=kn(zk)
        prediction=blur_image(latent,kernel)
        r=(prediction.shape[-2]-h)//2;c=(prediction.shape[-1]-w)//2
        prediction=prediction[...,r:r+h,c:c+w]
        objective=loss_for_step(prediction,target,step)
        if not torch.isfinite(objective):
            raise FloatingPointError(f'nonfinite loss at step {step}')
        objective.backward();optimizer.step()
        trace.append((step,float(objective.detach()),*rates))
    if device.startswith('cuda'):torch.cuda.synchronize()
    return ReconstructionResult(latent.detach().cpu().numpy()[0,0,25:25+h,25:25+w].copy(),
                                kernel.detach().cpu().numpy()[0,0].copy(),np.asarray(trace),
                                time.perf_counter()-start)


