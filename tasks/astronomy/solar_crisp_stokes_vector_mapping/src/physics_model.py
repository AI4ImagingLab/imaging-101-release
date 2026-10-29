"""Explicit polarized Milne–Eddington transfer, Zeeman patterns and responses.
MIT, adapted from J. de la Cruz Rodriguez's pyMilne (ISP-SU2020).
Forward derivatives propagate analytic Voigt identities and task equations locally.
"""
import numpy as np

class Dual:
    """Nine-component forward differentials; task algebra remains inspectable."""
    __array_priority__=1000
    def __init__(self,value,derivative=None):
        self.v=np.asarray(value,dtype='f8')
        self.d=0. if derivative is None else derivative
    def __add__(self,other):
        other=as_dual(other); return Dual(self.v+other.v,self.d+other.d)
    __radd__=__add__
    def __neg__(self): return Dual(-self.v,-self.d)
    def __sub__(self,other): return self+-as_dual(other)
    def __rsub__(self,other): return as_dual(other)+-self
    def __mul__(self,other):
        other=as_dual(other); return Dual(self.v*other.v,self.d*other.v[...,None]+other.d*self.v[...,None])
    __rmul__=__mul__
    def __truediv__(self,other):
        other=as_dual(other); return Dual(self.v/other.v,(self.d*other.v[...,None]-other.d*self.v[...,None])/other.v[...,None]**2)
    def __rtruediv__(self,other): return as_dual(other)/self
    def __pow__(self,power): return Dual(self.v**power,power*self.v[...,None]**(power-1)*self.d)

def as_dual(x): return x if isinstance(x,Dual) else Dual(x)
def sin(x): return Dual(np.sin(x.v),np.cos(x.v)[...,None]*x.d)
def cos(x): return Dual(np.cos(x.v),-np.sin(x.v)[...,None]*x.d)

def humlicek(a,v):
    """Original four-region Humlicek complex Voigt approximation, H+i2F."""
    a,v=np.broadcast_arrays(a,v); z=a-1j*v; u=z*z; s=abs(v)+a; out=np.empty_like(z)
    m=s>=15; out[m]=z[m]*.5641896/(.5+u[m])
    m=(s<15)&(s>=5.5); t=z[m]; q=u[m]; out[m]=t*(1.410474+q*.5641896)/(.75+q*(3+q))
    m=(s<5.5)&(a>=.195*abs(v)-.176); t=z[m]
    out[m]=(16.4955+t*(20.20933+t*(11.96482+t*(3.778987+t*.5642236))))/(16.4955+t*(38.82363+t*(39.27121+t*(21.69274+t*(6.699398+t)))))
    m=(s<5.5)&(a<.195*abs(v)-.176); t=z[m]; q=u[m]
    numerator=t*(36183.31-q*(3321.9905-q*(1540.787-q*(219.0313-q*(35.76683-q*(1.320522-q*.56419))))))
    denominator=32066.6-q*(24322.84-q*(9022.228-q*(2186.181-q*(364.2191-q*(61.57037-q*(1.841439-q))))))
    out[m]=np.exp(q)-numerator/denominator
    return out

def profile(damping,frequency):
    """Voigt/Faraday values and original analytic response identities."""
    w=humlicek(damping.v,frequency.v); h=w.real; f=w.imag
    dh=2*damping.v*f-2*frequency.v*h
    df=2*(1/1.7724538509055159-damping.v*h-frequency.v*f)
    return Dual(h,dh[...,None]*frequency.d-df[...,None]*damping.d),Dual(f,df[...,None]*frequency.d+dh[...,None]*damping.d)

def zeeman_components(j1,j2,g1,g2,wavelength):
    """Landi transition strengths and signed wavelength split per Gauss."""
    factor=4.80320441e-10/(4e8*np.pi*9.10938188e-28*2.99792458e10**2)
    components=[]; delta=int(j2-j1)
    for iup in range(1,int(2*j2)+2):
        upper=j2+1-iup
        for pol in range(1,4):
            lower=upper-2+pol
            if abs(lower)>j1: continue
            if delta==0:
                strength=([1.5*(j1-lower)*(j1+lower+1),3*lower*lower,1.5*(j1+lower)*(j1-lower+1)][pol-1])/(j1*(j1+1)*(2*j1+1))
            elif delta==-1:
                strength=([1.5*(j1-lower)*(j1-lower-1),3*(j1-lower)*(j1+lower),1.5*(j1+lower)*(j1+lower-1)][pol-1])/(j1*(2*j1-1)*(2*j1+1))
            elif delta==1:
                strength=([1.5*(j1+lower+1)*(j1+lower+2),3*(j1-lower+1)*(j1+lower+1),1.5*(j1-lower+1)*(j1-lower+2)][pol-1])/((j1+1)*(2*j1+1)*(2*j1+3))
            else: strength=0.
            if strength<=1e-5: continue
            splitting=-(g2*upper-g1*lower)*factor*wavelength*wavelength
            components.append((pol-1,strength,splitting))
    return components

LINES=[(6301.4995,1.,zeeman_components(2,2,1.84,1.50,6301.4995)),(6302.4931,10**(-1.160+.718),zeeman_components(1,0,2.49,0,6302.4931))]

def degrade(array,psf):
    """Original edge extension and FFT cross-correlation, spectral axis last."""
    n=array.shape[-1]; size=len(psf); padding=size if size%2==0 else size-1; length=n+padding
    padded=np.concatenate((array,np.repeat(array[...,-1:],size//2,axis=-1),np.repeat(array[...,:1],length-n-size//2,axis=-1)),axis=-1)
    kernel=np.zeros(length); kernel[:size]=psf/psf.sum(); kernel=np.roll(kernel,-size//2 if size%2==0 else -(size//2))
    return np.fft.irfft(np.fft.rfft(padded,axis=-1)*np.conj(np.fft.rfft(kernel)),n=length,axis=-1)[...,:n]

def synthesize_response(parameters,regions,psfs,mu=.93,with_response=True):
    """Parameters(N,9) -> spectra(N,4,W), responses(N,9,4,W), float64."""
    parameters=np.asarray(parameters,dtype='f8').reshape(-1,9); n=len(parameters)
    m=[]
    for k in range(9):
        derivative=np.zeros((n,1,9 if with_response else 0))
        if with_response: derivative[:,:,k]=1
        m.append(Dual(parameters[:,k,None],derivative))
    field,inc,azi,velocity,width,opacity,damping,s0,s1=m
    wave=np.concatenate(regions)[None,:]; nw=wave.shape[-1]
    ci=cos(inc); si2=sin(inc)**2; ca=cos(2*azi); sa=sin(2*azi)
    ki=Dual(np.ones((n,nw))); kq=Dual(np.zeros((n,nw))); ku=Dual(np.zeros((n,nw))); kv=Dual(np.zeros((n,nw))); fq=Dual(np.zeros((n,nw))); fu=Dual(np.zeros((n,nw))); fv=Dual(np.zeros((n,nw)))
    for center,gf,components in LINES:
        hs=[Dual(np.zeros((n,nw))) for _ in range(3)]; fs=[Dual(np.zeros((n,nw))) for _ in range(3)]
        shift=center*velocity/299792.458
        for pol,strength,split in components:
            frequency=(wave-center-shift-split*field)/width
            h,f=profile(damping,frequency); hs[pol]=hs[pol]+strength*h; fs[pol]=fs[pol]+strength*f
        eta=gf*opacity/2; hd=hs[1]-(hs[0]+hs[2])/2; fd=fs[1]-(fs[0]+fs[2])/2
        ki=ki+eta*(hs[1]*si2+(hs[0]+hs[2])*(1+ci**2)/2)
        kq=kq+eta*hd*si2*ca; ku=ku+eta*hd*si2*sa; kv=kv+eta*(hs[2]-hs[0])*ci
        fq=fq+eta*fd*si2*ca; fu=fu+eta*fd*si2*sa; fv=fv+eta*(fs[2]-fs[0])*ci
    gp3=ki**2+fq**2+fu**2+fv**2
    gp1=gp3-kq**2-ku**2-kv**2; gp2=kq*fq+ku*fu+kv*fv
    det=ki**2*gp1-gp2**2; factor=s1*mu/det
    gp4=ki**2*kq+ki*(kv*fu-ku*fv); gp5=ki**2*ku+ki*(kq*fv-kv*fq); gp6=ki**2*kv+ki*(ku*fq-kq*fu)
    stokes=[s0+factor*ki*gp3,-factor*(gp4+fq*gp2),-factor*(gp5+fu*gp2),-factor*(gp6+fv*gp2)]
    spectra=np.stack([s.v for s in stokes],axis=1); response=np.stack([s.d for s in stokes],axis=1).transpose(0,3,1,2)
    start=0
    for region,psf in zip(regions,psfs):
        stop=start+len(region)
        spectra[...,start:stop]=degrade(spectra[...,start:stop],psf)
        response[...,start:stop]=degrade(response[...,start:stop],psf)
        start=stop
    return spectra,response
