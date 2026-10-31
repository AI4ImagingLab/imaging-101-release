"""Public objective contract controls; no solver imports."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_stored_fit_and_float32_roundoff(tmp_path):
    with np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as f:
        fields={k:f[k] for k in f.files}
    for dtype in (np.float64,np.float32):
        path=tmp_path/'prediction.npz';np.savez(path,**{k:v.astype(dtype) for k,v in fields.items()})
        report,code=judge.evaluate(path)
        assert code==0, report

def test_one_bad_voxel_cannot_hide_in_global_average(tmp_path):
    with np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as f:fields={k:f[k].copy() for k in f.files}
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:mask=f['observation_mask']
    index=tuple(np.argwhere(mask)[0])
    for value in fields.values():value[index]=0
    path=tmp_path/'prediction.npz';np.savez(path,**fields)
    report,code=judge.evaluate(path)
    assert code==1,report
    assert report['metrics']['maximum_normalized_mse_excess']>1e-6


def test_independent_lower_loss_solution_is_accepted(tmp_path):
    from scipy.optimize import least_squares
    from scipy.special import erf
    with np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as f:fields={k:f[k].copy() for k in f.files}
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:
        mask=f['observation_mask'];b=f['bvalues'].ravel();data=f['diffusion_signal']
    improvements=[]
    for ix in np.argwhere(mask)[::500]:
        index=tuple(ix); y=data[index].astype(float);s0=y[b==0].mean();positive=b>0
        def residual(z):
            v,d=z[0],z[1]*.00305
            def response(t):
                root=np.sqrt(np.maximum(b[positive]*(d-t),0));ratio=np.ones_like(root)
                np.divide(np.sqrt(np.pi)*erf(root),2*root,out=ratio,where=root!=0)
                return np.exp(-b[positive]*t)*ratio
            return (s0*(v*response(0)+(1-v)*response((1-v)*d))-y[positive])/s0
        start=np.clip([fields['intra_fraction'][index],fields['intrinsic_diffusivity'][index]/.00305],0,1)
        fitted=least_squares(residual,start,bounds=(0,1),gtol=1e-12,ftol=1e-12,xtol=1e-12)
        if np.sum(residual(fitted.x)**2)<np.sum(residual(start)**2):
            v,d=fitted.x[0],fitted.x[1]*.00305
            improvements.append(float(np.sum(residual(start)**2)-np.sum(residual(fitted.x)**2)))
            fields['intra_fraction'][index]=v;fields['intrinsic_diffusivity'][index]=d
            fields['extra_transverse_diffusivity'][index]=(1-v)*d
            fields['extra_mean_diffusivity'][index]=(1-2*v/3)*d
    assert improvements,'No distinct improved fit was established'
    path=tmp_path/'independent.npz';np.savez(path,**fields)
    report,code=judge.evaluate(path)
    assert code==0,report
