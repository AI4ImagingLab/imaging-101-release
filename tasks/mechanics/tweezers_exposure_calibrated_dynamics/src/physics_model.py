"""Explicit exposure-averaged Langevin statistics, gamma likelihood and derivatives."""
import numpy as np
from scipy.special import gammaln
from .preprocessing import FS,KT
def prediction(x,parameters,kind='psd',exposure=True):
    g,k=parameters[:2];e=parameters[2] if len(parameters)==3 else 0.;tc=g/k
    if kind=='psd':
        if exposure:
            return 2*KT*tc/k*(1+2*tc*FS*np.sin(np.pi*x/FS)**2*np.sinh((1/tc)/FS)/(np.cos(2*np.pi*x/FS)-np.cosh((1/tc)/FS)))+e**2/FS
        return KT/(k*FS)*np.sinh((1/tc)/FS)/(np.cosh((1/tc)/FS)-np.cos(2*np.pi*x/FS))+e**2/FS
    if kind=='av':return 2*KT*tc/(k*x)*(1+2*tc/x*np.exp(-x/tc)-tc/(2*x)*np.exp(-2*x/tc)-3*tc/(2*x))+e**2/(FS*x)
    raise ValueError('kind must be psd or av')
def negative_log_likelihood(parameters,data,kind='psd',exposure=True):
    mu=prediction(data['x'],parameters,kind,exposure);a=data['shape'];y=data['y']
    if np.any(mu<=0) or not np.isfinite(mu).all():return np.inf
    scale=mu/a
    return float(-np.sum((a-1)*np.log(y/scale)-y/scale-gammaln(a)-np.log(scale)))
def derivatives(parameters,data,kind='psd'):
    """General tensor autodifferentiation of explicit same equations; no domain API."""
    import torch
    p=torch.tensor(parameters,dtype=torch.float64,requires_grad=True)
    x=torch.tensor(data['x'],dtype=torch.float64);y=torch.tensor(data['y'],dtype=torch.float64);a=torch.tensor(data['shape'],dtype=torch.float64)
    sin2=torch.tensor(np.sin(np.pi*data['x']/FS)**2,dtype=torch.float64)
    cos=torch.tensor(np.cos(2*np.pi*data['x']/FS),dtype=torch.float64)
    def mean(v):
        g,k=v[:2];e=v[2] if len(v)==3 else 0.;tc=g/k
        if kind=='psd':mu=2*KT*tc/k*(1+2*tc*FS*sin2*torch.sinh((1/tc)/FS)/(cos-torch.cosh((1/tc)/FS)))+e**2/FS
        else:mu=2*KT*tc/(k*x)*(1+2*tc/x*torch.exp(-x/tc)-tc/(2*x)*torch.exp(-2*x/tc)-3*tc/(2*x))+e**2/(FS*x)
        return mu
    def loss(v):
        mu=mean(v);scale=mu/a
        return -torch.sum((a-1)*torch.log(y/scale)-y/scale-torch.lgamma(a)-torch.log(scale))
    grad=_numpy_gradient(parameters,data,kind)
    hess=torch.autograd.functional.hessian(loss,p)
    return grad,hess.detach().numpy()

# Explicit reverse chain rules retain NumPy reduction order near cancellation.
# This small diagnostic graph contains no domain library or optimizer.
class _ChainNode:
    __array_priority__=1000
    def __init__(self,value,parents=()):self.v=value;self.parents=parents
    def binary(self,other,op,left,right):
        b=other.v if isinstance(other,_ChainNode) else other;a=self.v
        def reduce(x,g):return np.sum(g) if np.ndim(x)==0 and np.ndim(g)>0 else g
        parents=[(self,lambda g:reduce(a,left(g,a,b)))]
        if isinstance(other,_ChainNode):parents.append((other,lambda g:reduce(b,right(g,a,b))))
        return _ChainNode(op(a,b),parents)
    def __add__(self,b):return self.binary(b,lambda a,b:a+b,lambda g,a,b:g,lambda g,a,b:g)
    __radd__=__add__
    def __sub__(self,b):return self.binary(b,lambda a,b:a-b,lambda g,a,b:g,lambda g,a,b:-g)
    def __rsub__(self,a):return _ChainNode(a).__sub__(self)
    def __mul__(self,b):return self.binary(b,lambda a,b:a*b,lambda g,a,b:g*b,lambda g,a,b:g*a)
    __rmul__=__mul__
    def __truediv__(self,b):return self.binary(b,lambda a,b:a/b,lambda g,a,b:g/b,lambda g,a,b:-g*a/b**2)
    def __rtruediv__(self,a):return _ChainNode(a).__truediv__(self)
    def __pow__(self,b):return _ChainNode(self.v**b,[(self,lambda g:g*b*self.v**(b-1))])
    def __neg__(self):return _ChainNode(-self.v,[(self,lambda g:-g)])
    def sum(self):return _ChainNode(np.sum(self.v),[(self,lambda g:np.ones_like(self.v)*g)])
    def unary(self,f,df):return _ChainNode(f(self.v),[(self,lambda g:g*df(self.v))])
def _reverse(root,leaves):
    counts={};stack=[root]
    while stack:
        n=stack.pop()
        if n in counts:counts[n]+=1
        else:counts[n]=1;stack.extend(p for p,_ in n.parents)
    stack=[root];grads={root:1.}
    while stack:
        n=stack.pop()
        for p,vjp in n.parents:
            v=vjp(grads[n]);grads[p]=grads[p]+v if p in grads else v
            if counts[p]==1:stack.append(p)
            else:counts[p]-=1
    return np.array([grads[n] for n in leaves])
def _numpy_gradient(parameters,data,kind):
    p=[_ChainNode(v) for v in parameters];g,k=p[:2];e=p[2] if len(p)==3 else 0.;x=data['x'];a=data['shape'];y=data['y'];tc=g/k;fc=1./tc
    exp=lambda z:z.unary(np.exp,np.exp)
    sinh=lambda z:z.unary(np.sinh,np.cosh)
    cosh=lambda z:z.unary(np.cosh,np.sinh)
    log=lambda z:z.unary(np.log,lambda v:1/v)
    if kind=='psd':mu=2.*4.1*tc/k*(1.+2.*tc*400*np.sin(np.pi*x/400)**2*sinh(fc/400)/(np.cos(2*np.pi*x/400)-cosh(fc/400)))+e**2/400
    else:mu=2.*4.1*tc/(k*x)*(1.+2.*(tc/x)*exp(-x/tc)-(tc/(2.*x))*exp(-2.*x/tc)-3.*tc/(2.*x))+e**2/(400*x)
    scale=mu/a;z=y/scale
    gamma=z.unary(lambda v:(a-1)*np.log(v)-v-gammaln(a),lambda v:(a-v-1)/v)
    loss=-(gamma-log(scale)).sum()
    return _reverse(loss,p)
