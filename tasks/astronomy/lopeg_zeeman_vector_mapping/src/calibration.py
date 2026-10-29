"""Explicit selected ZDIpy mathematics, adapted from C. P. Folsom.
GPL-2.0; source revision 86d9f98c426a709f3298bab95fda5f6f5e65e111.
Only the bound full-vector, chi-square-target problem is supported.
"""
import numpy as np
def equivWidComp2(lineStr, meanEquivWidObs, setSynSpec, lineData):
    lineStr0 = lineData.str[0]
    nObs = 0
    meanEW = 0.0
    for spec in setSynSpec:
        scaleI = 1.0 - (1.0 - spec.IIc) * (lineStr / lineStr0)
        equivWidApprox = 0.0
        for i in range(spec.wl.shape[0] - 1):
            equivWidApprox += (1.0 - scaleI[i]) * (spec.wl[i + 1] - spec.wl[i])
        equivWidApprox += (1.0 - scaleI[-1]) * (spec.wl[-1] - spec.wl[-2])
        meanEW += equivWidApprox
        nObs += 1
    meanEW /= float(nObs)
    diffEW = np.abs(meanEquivWidObs - meanEW)
    return diffEW

def fitLineStrength(meanEquivWidObs, par, listGridView, vecMagCart, dMagCart0, briMap, lineData, wlSynSet, verbose=1):
    from scipy.optimize import minimize_scalar
    from . import physics_model as lineprofile
    if verbose == 1:
        print('fitting line strength (by equivalent width)')
    nObs = 0
    setSynSpec = []
    for phase in par.cycleList:
        spec = lineprofile.diskIntProfAndDeriv(listGridView[nObs], vecMagCart, dMagCart0, briMap, lineData, par.velEq, wlSynSet[nObs], 0, 0)
        spec.convolveIGnumpy(par.instrumentRes)
        setSynSpec += [spec]
        nObs += 1
    fitLineStr = minimize_scalar(equivWidComp2, bracket=(lineData.str[0] * 0.01, lineData.str[0], lineData.str[0] * 100.0), method='brent', tol=1e-05, args=(meanEquivWidObs, setSynSpec, lineData))
    if verbose == 1:
        ewidth = equivWidComp2(fitLineStr.x, 0, setSynSpec, lineData)
        print('best match line strength is {:f} (ew {:f} )'.format(fitLineStr.x, ewidth))
    lineData.str[0] = fitLineStr.x
