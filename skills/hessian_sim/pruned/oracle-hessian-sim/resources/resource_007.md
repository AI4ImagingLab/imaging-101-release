# Hessian split-Bregman (mu=150, sigma_z=1.0, n_iter=100, lamda=mu*20)

Axis 0=z/t, 1=y, 2=x. Periodic BC.
fdiff(x,dim)=roll(x,-1,dim)-x ; bdiff(x,dim)=x-roll(x,1,dim)

## Denominator kernels (sizex=(nz,sy,sx)); Frefft = sum fftn(k)*conj(fftn(k)) with weights
k1[0,0,:3]=[1,-2,1]         w=1        (Dxx)
k2[0,:3,0]=[1,-2,1]         w=1        (Dyy)
k3[:3,0,0]=[1,-2,1]         w=zbei^2   (Dzz)
k4[0,:2,:2]=[[1,-1],[-1,1]] w=2        (Dxy)
k5[:2,0,:2]=[[1,-1],[-1,1]] w=2*zbei   (Dxz)
k6[:2,:2,0]=[[1,-1],[-1,1]] w=2*zbei   (Dyz)
divide = real((mu/lamda) + Frefft)

## Iteration (frac carries across iterations)
frac = (mu/lamda)*y
for it in range(n_iter):
    x = ifftn(fftn(frac)/divide) if it>0 else ifftn(fftn(frac)/(mu/lamda)); take real
    frac = (mu/lamda)*y
    # Dxx
    u=bdiff(fdiff(x,2),2); d=shrink(u+b0,1/lamda); b0+=u-d; frac+=bdiff(fdiff(d-b0,2),2)
    # Dyy
    u=bdiff(fdiff(x,1),1); d=shrink(u+b1,1/lamda); b1+=u-d; frac+=bdiff(fdiff(d-b1,1),1)
    # Dzz
    u=bdiff(fdiff(x,0),0); d=shrink(u+b2,1/lamda); b2+=u-d; frac+=zbei^2*bdiff(fdiff(d-b2,0),0)
    # Dxy
    u=fdiff(fdiff(x,2),1); d=shrink(u+b3,1/lamda); b3+=u-d; frac+=2*bdiff(bdiff(d-b3,1),2)
    # Dxz
    u=fdiff(fdiff(x,2),0); d=shrink(u+b4,1/lamda); b4+=u-d; frac+=2*zbei*bdiff(bdiff(d-b4,0),2)
    # Dyz
    u=fdiff(fdiff(x,1),0); d=shrink(u+b5,1/lamda); b5+=u-d; frac+=2*zbei*bdiff(bdiff(d-b5,0),1)
shrink(v,t) = max(|v|-t,0)*sign(v)

## Finish
x[x<0]=0; x = x[:nz_orig]*ymax
