"""Standard CPU entry point for joint variational motion deblatting."""
import argparse
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data, OBJECT_SHAPE
from .solvers import Params, estimateFMH
from .physics_model import project2pyramid_exact


def polish_feasible_scales(arrays,observation,background):
    """Solve the four-variable convex intensity/opacity scale subproblem exactly."""
    from scipy.optimize import minimize,LinearConstraint
    from .physics_model import forward,createRnMatrix
    F=arrays['reconstruction'][0];M=arrays['mask'][0];H=arrays['kernel'][0]
    blur_f=forward(np.zeros_like(background),H,F,M)
    blur_m=1-forward(np.ones_like(background),H,np.zeros_like(F),M)[...,0]
    design=np.zeros((observation.size,4))
    for c in range(3):
        column=np.zeros_like(observation);column[...,c]=blur_f[...,c];design[:,c]=column.ravel()
    design[:,3]=(-background*blur_m[...,None]).ravel();target=(observation-background).ravel()
    variation=[]
    for field in [F[...,c] for c in range(3)]+[M]:
        u=np.pad(field,((1,1),(1,1)))
        variation.append(np.hypot(np.diff(u,axis=1)[1:,:],np.diff(u,axis=0)[:,1:]).sum())
    rotation=createRnMatrix(M.shape);m=M.flatten('F');symmetry=np.linalg.norm(rotation@m-m)**2
    normal=design.T@design;normal[3,3]+=.01*symmetry
    linear=-design.T@target+2**(-12)*np.asarray(variation)
    constraints=np.zeros((3,4))
    for c in range(3):
        positive=F[...,c]>0;limit=np.min(M[positive]/F[...,c][positive]) if positive.any() else 1e100
        constraints[c,c]=1.;constraints[c,3]=-limit
    objective=lambda a: .5*float(a@normal@a)+float(linear@a)
    solution=minimize(objective,np.ones(4),jac=lambda a:normal@a+linear,method='SLSQP',bounds=[(0,None)]*3+[(0,1/M.max())],constraints=[LinearConstraint(constraints,-np.inf,0)],options={'ftol':1e-13,'maxiter':200})
    if not solution.success:raise RuntimeError('Feasible scale polish failed: '+solution.message)
    a=solution.x
    if objective(a)>objective(np.ones(4))+1e-12:raise RuntimeError('Scale polish increased the conditional objective')
    F=F*a[:3];M=M*a[3]
    spectrum=np.fft.fft2(H)
    def tv_value_gradient(u):
        extended=np.pad(u,((1,1),(1,1)))
        dx=np.diff(extended,axis=1)[1:,:];dy=np.diff(extended,axis=0)[:,1:]
        norm=np.hypot(dx,dy);px=np.divide(dx,norm,out=np.zeros_like(dx),where=norm>0);py=np.divide(dy,norm,out=np.zeros_like(dy),where=norm>0)
        gradient=px[:36,:36]-px[:36,1:]+py[:36,:36]-py[1:,:36]
        return float(norm.sum()),gradient
    def loss_and_gradient(colors,opacity):
        residual=forward(background,H,colors,opacity)-observation
        gf=np.stack([np.roll(np.fft.ifft2(spectrum.conj()*np.fft.fft2(residual[...,c])).real,(18,18),axis=(0,1))[:36,:36] for c in range(3)],axis=-1)
        gm=-np.roll(np.fft.ifft2(spectrum.conj()*np.fft.fft2(np.sum(background*residual,axis=-1))).real,(18,18),axis=(0,1))[:36,:36]
        value=.5*float(np.sum(residual**2))
        for c in range(3):
            v,g=tv_value_gradient(colors[...,c]);value+=2**(-12)*v;gf[...,c]+=2**(-12)*g
        v,g=tv_value_gradient(opacity);value+=2**(-12)*v;gm+=2**(-12)*g
        m=opacity.flatten('F');diff=rotation@m-m;value+=.005*float(diff@diff);gm+=(.01*(rotation.T@diff-diff)).reshape(36,36,order='F')
        return value,gf,gm
    for iteration in range(5):
        value,gf,gm=loss_and_gradient(F,M);step=.01
        for attempt in range(24):
            projected_m,projected_f=project2pyramid_exact((M-step*gm).ravel(),(F-step*gf).reshape(-1,3),1)
            trial_m=projected_m.reshape(36,36);trial_f=projected_f.reshape(36,36,3)
            trial=loss_and_gradient(trial_f,trial_m)[0]
            if trial<value-1e-14:
                F,M=trial_f,trial_m;break
            step*=.5
        else:break
    return {'reconstruction':F[None],'mask':M[None],'kernel':H[None]}


def reconstruct(observation, background, projection_mode='exact'):
    """Return appearance, opacity and exposure kernel from one measured crop."""
    parameters = Params()
    if projection_mode not in ('exact', 'upstream_cyclic'):
        raise ValueError('Unknown projection mode.')
    parameters.projection_mode = projection_mode
    parameters.verbose = False
    parameters.visualize = False
    kernel, appearance, mask = estimateFMH(observation, background,
                                         np.ones(OBJECT_SHAPE), params=parameters)
    if projection_mode == 'exact':
        opacity,colors=project2pyramid_exact(mask.ravel(),appearance.reshape(-1,3),1)
        mask=opacity.reshape(OBJECT_SHAPE);appearance=colors.reshape(OBJECT_SHAPE+(3,))
        kernel=np.maximum(kernel,0);kernel/=kernel.sum()
    arrays={'reconstruction': appearance[None], 'mask': mask[None], 'kernel': kernel[None]}
    return polish_feasible_scales(arrays,observation,background) if projection_mode=='exact' else arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw_data.npz')
    parser.add_argument('--output', default='output/reconstruction.npz')
    parser.add_argument('--projection', choices=('exact','upstream_cyclic'), default='exact')
    args = parser.parse_args()
    start = time.perf_counter()
    result = reconstruct(*load_data(args.data), projection_mode=args.projection)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f'Reconstructed all three unknowns in {time.perf_counter()-start:.3f} s.')


if __name__ == '__main__':
    main()
