# Adapted from refnx; Copyright 2011-2026 A. R. J. Nelson, ANSTO.
# BSD-3-Clause; see task LICENSE. Explicit local mathematics, no refnx imports.
import numpy as np
from scipy.special import erf
TINY=1e-30
INITIAL=np.array([1.,3e-6,30.,3.,250.,2.,3.,3.])
BOUNDS=np.array([[.6,1.2],[1e-9,9e-6],[15.,50.],[1.,15.],[200.,300.],[.1,3.],[1.,15.],[1.,15.]])
Z=np.linspace(-30.,380.,821)
def slabs(p):
    """Map eight parameters to rows [thickness, real SLD, imaginary SLD, roughness]."""
    return np.array([[0.,2.07,0.,0.],[p[2],3.47,0.,p[3]],[p[4],p[5],0.,p[6]],[0.,6.36,0.,p[7]]])
def abeles(
    q,
    layers,
    scale: float | None = 1.0,
    bkg: float | None = 0,
    threads: int | None = 0,
) -> np.array:
    """
    Abeles matrix formalism for calculating reflectivity from a stratified
    medium.

    Parameters
    ----------
    q : array_like
        the q values required for the calculation.
        Q = 4 * Pi / lambda * sin(omega).
        Units = Angstrom**-1
    layers : np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers:

        - layers[0, 1]
           SLD of fronting (/1e-6 Angstrom**-2)
        - layers[0, 2]
           iSLD of fronting (/1e-6 Angstrom**-2)
        - layers[N, 0]
           thickness of layer N
        - layers[N, 1]
           SLD of layer N (/1e-6 Angstrom**-2)
        - layers[N, 2]
           iSLD of layer N (/1e-6 Angstrom**-2)
        - layers[N, 3]
           roughness between layer N-1/N
        - layers[-1, 1]
           SLD of backing (/1e-6 Angstrom**-2)
        - layers[-1, 2]
           iSLD of backing (/1e-6 Angstrom**-2)
        - layers[-1, 3]
           roughness between backing and last layer

    scale : float
        Multiply all reflectivities by this value.
    bkg : float
        Linear background to be added to all reflectivities
    threads : int, optional
        <THIS OPTION IS CURRENTLY IGNORED>

    Returns
    -------
    Reflectivity : np.ndarray
        Calculated reflectivity values for each q value.
    """
    qvals = np.asarray(q).astype(float, copy=False)
    flatq = qvals.ravel()

    nlayers = layers.shape[0] - 2
    npnts = flatq.size

    kn = np.zeros((npnts, nlayers + 2), np.complex128)
    mi00 = np.ones((npnts, nlayers + 1), np.complex128)

    sld = np.zeros(nlayers + 2, np.complex128)

    # addition of TINY is to ensure the correct branch cut
    # in the complex sqrt calculation of kn.
    sld[1:] += (
        (layers[1:, 1] - layers[0, 1]) + 1j * (np.abs(layers[1:, 2]) + TINY)
    ) * 1.0e-6

    # kn is a 2D array. Rows are Q points, columns are kn in a layer.
    # calculate wavevector in each layer, for each Q point.
    kn[:] = np.sqrt(flatq[:, np.newaxis] ** 2.0 / 4.0 - 4.0 * np.pi * sld)

    # reflectances for each layer
    # rj.shape = (npnts, nlayers + 1)
    rj = kn[:, :-1] - kn[:, 1:]
    rj /= kn[:, :-1] + kn[:, 1:]
    rj *= np.exp(-2.0 * kn[:, :-1] * kn[:, 1:] * layers[1:, 3] ** 2)

    # characteristic matrices for each layer
    # miNN.shape = (npnts, nlayers + 1)
    if nlayers:
        mi00[:, 1:] = np.exp(kn[:, 1:-1] * 1j * np.fabs(layers[1:-1, 0]))
    mi11 = 1.0 / mi00
    mi10 = rj * mi00
    mi01 = rj * mi11

    # initialise matrix total
    mrtot00 = mi00[:, 0]
    mrtot01 = mi01[:, 0]
    mrtot10 = mi10[:, 0]
    mrtot11 = mi11[:, 0]

    # propagate characteristic matrices
    for idx in range(1, nlayers + 1):
        # matrix multiply mrtot by characteristic matrix
        p0 = mrtot00 * mi00[:, idx] + mrtot10 * mi01[:, idx]
        p1 = mrtot00 * mi10[:, idx] + mrtot10 * mi11[:, idx]
        mrtot00 = p0
        mrtot10 = p1

        p0 = mrtot01 * mi00[:, idx] + mrtot11 * mi01[:, idx]
        p1 = mrtot01 * mi10[:, idx] + mrtot11 * mi11[:, idx]

        mrtot01 = p0
        mrtot11 = p1

    r = mrtot01 / mrtot00
    reflectivity = r * np.conj(r)
    reflectivity *= scale
    reflectivity += bkg
    return np.real(np.reshape(reflectivity, qvals.shape))
def forward(p,q,q_fwhm):
    """Resolution-smeared intensity with upstream 17-point, +/-3.5 sigma quadrature."""
    x,w=np.polynomial.legendre.leggauss(17)
    sigma=q_fwhm/(2*np.sqrt(2*np.log(2.)))
    samples=q[:,None]+3.5*sigma[:,None]*x
    r=abeles(samples,slabs(p))
    smeared=3.5*np.sum(r*(w*np.exp(-.5*(3.5*x)**2)/np.sqrt(2*np.pi)),axis=-1)
    return p[0]*smeared+p[1]

import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def evaluate(path):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((HERE/'metrics.json').read_text())
        for key in ('profile_nrmse_boundary','parameter_error_boundary','log_chisqr_boundary'):
            if not np.isfinite(cfg[key]) or cfg[key]<=0: raise ValueError('undefined metric configuration')
        reference_path=HERE/'ground_truth.npz' if (HERE/'ground_truth.npz').exists() else HERE/'baseline_reference.npz'
        with np.load(reference_path,allow_pickle=False) as f: ref={k:f[k] for k in ('reconstruction','parameters')}
        for k,shape in [('reconstruction',(1,821)),('parameters',(1,8))]:
            if ref[k].shape!=shape or ref[k].dtype!=np.float64 or not np.isfinite(ref[k]).all(): raise ValueError('malformed evaluator reference')
        if np.ptp(ref['reconstruction'])<=0: raise ValueError('degenerate evaluator reference')
        with np.load(path,allow_pickle=False) as f:
            x={}
            for k,shape in [('reconstruction',(1,821)),('parameters',(1,8))]:
                if k not in f: raise ValueError('missing required output '+k)
                x[k]=f[k]
                if x[k].shape!=shape or x[k].dtype!=np.float64 or not np.isfinite(x[k]).all(): raise ValueError('invalid shape/dtype/finiteness: '+k)
        p=x['parameters'][0]
        if np.any(p<BOUNDS[:,0]) or np.any(p>BOUNDS[:,1]):
            chisq=1e100
        else:
            with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as d:
                q,r,e,dq=(d[k][0] for k in ('q','reflectivity','standard_error','q_fwhm'))
            pred=forward(p,q,dq)
            chisq=float(np.sum(((np.log10(r)-np.log10(pred))/(e/r/np.log(10)))**2))
        m=dict(profile_nrmse=float(np.sqrt(np.mean((x['reconstruction']-ref['reconstruction'])**2))/np.ptp(ref['reconstruction'])),parameter_error=float(np.max(np.abs(x['parameters']-ref['parameters'])/(BOUNDS[:,1]-BOUNDS[:,0]))),log_chisqr=chisq)
        threshold={k:dict(comparison='<=',boundary=cfg[k+'_boundary']) for k in m}
        if not all(np.isfinite(v) for v in m.values()): raise ValueError('undefined metric')
        passed=all(v<=threshold[k]['boundary'] for k,v in m.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=m,thresholds=threshold)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    report,code=evaluate(a.prediction)
    try:
        target=Path(a.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc: print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__': sys.exit(main())
