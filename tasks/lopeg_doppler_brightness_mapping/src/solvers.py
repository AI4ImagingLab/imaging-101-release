"""Explicit selected ZDIpy mathematics, adapted from C. P. Folsom.
GPL-2.0; source revision 86d9f98c426a709f3298bab95fda5f6f5e65e111.
Only the bound full-vector, chi-square-target problem is supported.
"""
import numpy as np
from scipy import linalg
def mem_iter(n1, n2, ntot, Img, Data, Fmodel, sig2, Resp, weights, defImg, defIpm, ffIMax, targetAim, fixEntropy):
    """
    Maximum entropy method (MEM) image reconstruction, using the algorithm
    of Skilling & Bryan (1984, MNRAS 211, 111).
    This allows for different forms of entropy to be applied to different parts of
    the image vector, for simultainous magnetic and brightness mapping.

    :param n1: use standard image entropy (+ve, no upper limit) for the first n1 elements of Img.
    :param n2: use filling factor entropy (+ve, limiting upper value) for the n1:n2 elements of Img.
      Filling factor entropy follows A. Collier Cameron (1992, in Surface Inhomogeneities
      on Late-Type Stars, p. 33) and Y.C. Unruh & A. Collier Cameron (1995, MNRAS, 273, 1) Eq 4.
    :param ntot: total number of elements in the image vector Img.
      Positive/negative (magnetic) entropy (allows -ve & +ve, trends towards 0, no upper/lower limits)
      is used for the n2:ntot elements.  This form of entropy is from 
      Hobson & Lasenby (1998, MNRAS, 298, 905) (e.g. Eq 8)
    :param Img: array of current image values (ntot long), corresponds to f in Skilling and Bryan (Eq 1).
    :param Data: array of data values the model is fit to (Eq 2).
    :param Fmodel: array of model, i.e. simulated data, calculated from the current input image Img.
      (Same dimensions as Data, and for the same points.)  For ZDI this is the model spectrum.
    :param sig2: array standard deviations (1 sigma errors) squared, for data values in Data.
      (Same dimensions as Data.)
    :param Resp: response matrix, derivatives of the model (Fmodel) with respect to the image (Img).
      That is, each element of the Resp is defined as R(k,j) = dF(k)/dI(j).
      (Dimensions of [Data.shape, Img.shape])
    :param weights: array of additional weights, which are multiplied with the entropy terms
      for each image pixel.  (Same Dimension as Img)
      This is debatable from a theoretical stand point, since it would be more elegant
      and more in keeping with Bayesian statistics to include this in the default values
      for each pixel.  That would include per-pixel weights properly as a prior.  However in testing
      a simple multiplicative value was effective, and is more consistent with other ZDI codes. 
    :param defImg: default image intensity (Eq 6), for standard entropy.
      Used for the Img[0:n1] part of the image.  For filling factor entropy (Img[n1:n2] part of the image)
      this is also used as the default value (e.g. m in Eq 4 of Unruh & Collier Cameron (1995)).
    :param defIpm: "default" value for the positive/negative (magnetic) entropy.
      Actually controls the slope/change in slope in the entropy curve, the values will default to 0.
      See Hobson & Lasenby (1998), this for their eq 8,  m_f and m_g, when m_f = m_g = defIpm.
      Applies to Img[n2:nmax]
    :param ffIMax: maximum image value allowed for filling factor entropy as in, e.g.,
      Eq 4 of Unruh & Collier Cameron (1995) but replacing their 1 with this value
      (i.e. not limiting the 'filling factor' to strictly 1).
      Applies to Img[n1:n2]
    :param targetAim: the desired chi-squared (Eq 4) value to converge to,
      or the desired entropy to converge to.
    :param fixEntropy: A flag for whether to fit to target chi^2 or target entropy.
      Normaly this routine would fit to a target chi^2 level, but if this == 1
      the routine will fit to target entropy level.

    :return: entropy, chi2, test, Img, entStand, entFF, entMag
      * entropy: The entropy (Eq 6) of the input image.
      * chi2: The chi-squared (Eq 4) value for the input data and model.
      * test: The test statistic of Skilling and Bryan (Eq 37) for the input image, model, and data.
              This matches the returned entropy and chi2.
      * Img: Updated array of image values.
      * entStand: The entropy for the standard entropy part of the image.
      * entFF: The entropy for the 'filling factor' part of the image
      * entMag: The entropy for the positive/negative (magnetic) part of the image
    """
    maxDir = 10
    L_fac = 0.3
    C0, gradC = get_c_gradc(Data, Fmodel, sig2, Resp)
    S0, gradS, gradgradS, fsupi, Itot, entStand, entFF, entMag = get_s_grads(n1, n2, ntot, Img, weights, defImg, defIpm, ffIMax)
    test = get_test(gradC, gradS)
    edir, nedir, gamma = searchDir(ntot, maxDir, Resp, sig2, fsupi, gradC, gradS, gradgradS)
    Cmu, Smu = getCmuSmu(gradC, gradS, edir)
    L02 = getL0squared(L_fac, Itot)
    alphaMin = getAlphaMin(gamma)
    if fixEntropy == 1:
        raise ValueError('Only the selected chi-square target is supported')
    chiAim = targetAim
    Caimq = getCaimQuad(chiAim, C0, gamma, Cmu)
    xq = control(C0, gamma, Cmu, Smu, Caimq, L02, alphaMin)
    Img = updateImg(xq, edir, Img, n1, n2, ntot, ffIMax)
    entropy = S0
    chi2 = C0
    return (entropy, chi2, test, Img, entStand, entFF, entMag)

def get_c_gradc(Data, Fmodel, sig2, Resp):
    C0 = np.sum((Fmodel - Data) ** 2 / sig2)
    gradC = 2.0 * np.sum(Resp.T * (Fmodel - Data) / sig2, axis=1)
    return (C0, gradC)

def get_s_grads(n1, n2, ntot, Img, weights, defImg, defIpm, maxIff):
    gradS = np.zeros(ntot)
    gradgradS = np.zeros(ntot)
    fsupi = np.zeros(ntot)
    gradS[0:n1] = weights[0:n1] * (np.log(defImg) - np.log(Img[0:n1]))
    gradgradS[0:n1] = -weights[0:n1] / Img[0:n1]
    fsupi[0:n1] = -1.0 / gradgradS[0:n1]
    entStand = np.sum(-weights[0:n1] * (Img[0:n1] * (np.log(Img[0:n1] / defImg) - 1.0) + defImg))
    Itot = np.sum(weights[0:n1] * Img[0:n1])
    gradS[n1:n2] = weights[n1:n2] * (np.log(defImg / Img[n1:n2]) - np.log((maxIff - defImg) / (maxIff - Img[n1:n2])))
    gradgradS[n1:n2] = -weights[n1:n2] * maxIff / (Img[n1:n2] * (maxIff - Img[n1:n2]))
    fsupi[n1:n2] = -1.0 / gradgradS[n1:n2]
    entFF = np.sum(-weights[n1:n2] * (Img[n1:n2] * np.log(Img[n1:n2] / defImg) + (maxIff - Img[n1:n2]) * np.log((maxIff - Img[n1:n2]) / (maxIff - defImg))))
    Itot += np.sum(weights[n1:n2] * Img[n1:n2])
    tmpPsi = np.sqrt(Img[n2:ntot] ** 2 + 4.0 * defIpm ** 2)
    gradS[n2:ntot] = -weights[n2:ntot] * np.log((tmpPsi + Img[n2:ntot]) / (2.0 * defIpm))
    gradgradS[n2:ntot] = -weights[n2:ntot] / tmpPsi
    fsupi[n2:ntot] = -1.0 / gradgradS[n2:ntot]
    entMag = np.sum(weights[n2:ntot] * (tmpPsi - 2.0 * defIpm - Img[n2:ntot] * np.log((tmpPsi + Img[n2:ntot]) / (2.0 * defIpm))))
    Itot += np.sum(weights[n2:ntot] * np.maximum(np.abs(Img[n2:ntot]), defIpm))
    S0 = entStand + entFF + entMag
    return (S0, gradS, gradgradS, fsupi, Itot, entStand, entFF, entMag)

def searchDir(ntot, maxDir, Resp, sig2, fsupi, gradC, gradS, gradgradS):
    edir = np.zeros((ntot, maxDir))
    edir[:, 0] = gradC * fsupi
    sumNorm = np.dot(edir[:, 0], edir[:, 0])
    if sumNorm > 0.0:
        sumNorm = np.sqrt(sumNorm)
        err_gradCis0 = 0
    else:
        sumNorm = 1.0
        err_gradCis0 = 1
    edir[:, 0] /= sumNorm
    if maxDir == 1:
        return
    edir[:, 1] = gradS * fsupi
    sumNorm = np.dot(edir[:, 1], edir[:, 1])
    if sumNorm > 0.0:
        sumNorm = np.sqrt(sumNorm)
        err_gradSis0 = 0
    else:
        sumNorm = 1.0
        err_gradSis0 = 1
    edir[:, 1] /= sumNorm
    if (err_gradCis0 == 1) & (err_gradSis0 == 1):
        print('Error: f(gradS) and f(gradC) are both zero, problem not constrained')
        import sys
        sys.exit()
    for i in range(2, maxDir):
        tempDot = np.dot(Resp, edir[:, i - 2])
        edir[:, i] = 2.0 * fsupi * np.dot(Resp.T / sig2, tempDot)
        sumNorm = np.dot(edir[:, i], edir[:, i])
        if sumNorm > 0.0:
            edir[:, i] /= np.sqrt(sumNorm)
    edir, nedir, gamma = diagDir(edir, maxDir, gradgradS, sig2, Resp)
    return (edir, nedir, gamma)

def diagDir(edir, nedir, gradgradS, sig2, Resp):
    g = np.dot(-gradgradS * edir.T, edir)
    gamma, eiVec = linalg.eigh(g)
    maxGamma = np.max(np.abs(gamma))
    iok = gamma > 1e-08 * maxGamma
    edir = np.dot(edir, eiVec[:, iok]) / np.sqrt(gamma[iok])
    nedir = edir.shape[1]
    tmpRe = np.dot(Resp, edir)
    MM = 2.0 * np.dot(tmpRe.T / sig2, tmpRe)
    gammaM, eiVec = linalg.eigh(MM)
    edir = np.dot(edir, eiVec)
    return (edir, nedir, gammaM)

def getCmuSmu(gradC, gradS, edir):
    Cmu = np.dot(gradC, edir)
    Smu = np.dot(gradS, edir)
    return (Cmu, Smu)

def getL0squared(L_fac, Itot):
    L02 = L_fac * Itot
    return L02

def getAlphaMin(gamma):
    alphaMin = 0.0
    minGamma = np.min(gamma)
    if minGamma < 0.0:
        alphaMin = -minGamma
    return alphaMin

def getCaimQuad(chiAim, C0, gamma, Cmu):
    Cminq = C0 - 0.5 * np.sum(Cmu * Cmu / gamma)
    Caimq = 0.66666667 * Cminq + 0.33333333 * C0
    if chiAim > Caimq:
        Caimq = chiAim
    return Caimq

def chopDown(alpha, alphaLow):
    alphaHigh = alpha
    alpha = 0.5 * (alphaLow + alpha)
    return (alpha, alphaHigh)

def chopUp(alpha, alphaHigh):
    alphaLow = alpha
    if alphaHigh > 0.0:
        alpha = 0.5 * (alphaHigh + alpha)
    else:
        alpha = 2.0 * alpha + 0.1
    return (alpha, alphaLow)

def control(C0, gamma, Cmu, Smu, Caimq, L02, alphaMin):
    convTol = 1e-05
    if C0 < Caimq * 1.001:
        Caimr = Caimq * 1.001
    else:
        Caimr = Caimq
    P = Plow = 0.0
    Phigh = 0.0
    Pfinished = 0
    itera = 0
    iterP = 0
    while Pfinished == 0:
        alphaLow = alphaMin
        alphaHigh = -1.0
        alpha = alphaMin + 1.0
        afinished = 0
        while afinished == 0:
            asuccess = 0
            xqp = (alpha * Smu - Cmu) / (P + gamma + alpha)
            Cq = C0 + np.dot(Cmu, xqp) + 0.5 * np.einsum('i,i,i', gamma, xqp, xqp)
            Cqp = C0 + np.dot(Cmu, xqp) + 0.5 * np.einsum('i,i,i', P + gamma, xqp, xqp)
            L2 = np.dot(xqp, xqp)
            if (Cqp > C0) & (Cqp > Caimr):
                alpha, alphaHigh = chopDown(alpha, alphaLow)
            elif (Cq < Caimq) & (Cq < C0):
                alpha, alphaLow = chopUp(alpha, alphaHigh)
            elif L2 > L02:
                if Cqp < C0:
                    alpha, alphaLow = chopUp(alpha, alphaHigh)
                else:
                    alpha, alphaHigh = chopDown(alpha, alphaLow)
            else:
                asuccess = 1
                if Cq < Caimq:
                    alpha, alphaLow = chopUp(alpha, alphaHigh)
                else:
                    alpha, alphaHigh = chopDown(alpha, alphaLow)
            if (alphaHigh > 0.0) & (abs(alphaHigh - alphaLow) < convTol * alphaHigh + 1e-10):
                afinished = 1
            if alpha > 1e+20:
                afinished = 1
            itera += 1
        if asuccess != 1:
            P, Plow = chopUp(P, Phigh)
        elif (P == 0.0) | (abs(Phigh - Plow) < convTol * Phigh + 1e-10):
            Pfinished = 1
        else:
            P, Phigh = chopDown(P, Plow)
        if (asuccess == 0) & (P > 1e+20):
            Pfinished = 1
            print('P chop blow up encountered.')
        iterP += 1
    return xqp

def updateImg(xq, edir, Img, n1, n2, ntot, maxIff):
    Img += np.inner(xq, edir)
    intMod = np.where(Img[:n1] <= 0.0)
    Img[intMod] = 1e-06
    intMod = np.where(Img[n1:n2] <= 0.0)
    Img[intMod] = 1e-06 * maxIff
    intMod = np.where(Img[n1:n2] >= maxIff)
    Img[intMod] = maxIff * (1.0 - 1e-06)
    return Img

def get_test(gradC, gradS):
    mag_gradS = np.sqrt(np.sum(gradS ** 2))
    mag_gradC = np.sqrt(np.sum(gradC ** 2))
    if mag_gradS == 0.0:
        inv_mag_gradS = 0.0
    else:
        inv_mag_gradS = 1.0 / mag_gradS
    if mag_gradC == 0.0:
        inv_mag_gradC = 0.0
    else:
        inv_mag_gradC = 1.0 / mag_gradC
    test = 0.5 * np.sum((gradS * inv_mag_gradS - gradC * inv_mag_gradC) ** 2)
    return test

def packDataVector(obsSet, fitBri, fitMag):
    Data = np.empty(0)
    sig2 = np.empty(0)
    if fitBri == 1:
        for tmpObs in obsSet:
            Data = np.append(Data, tmpObs.specI)
            sig2 = np.append(sig2, tmpObs.specIsig ** 2)
    if fitMag == 1:
        for tmpObs in obsSet:
            Data = np.append(Data, tmpObs.specV)
            sig2 = np.append(sig2, tmpObs.specVsig ** 2)
    return (Data, sig2)

def packResponseMatrix(setSynSpec, nDataTot, npBriMap, magGeom, magGeomType, fitBri, fitMag):
    if fitBri == 1:
        nDataUsed = 0
        allModeldI = np.zeros([nDataTot, npBriMap])
        for spec in setSynSpec:
            allModeldI[nDataUsed:nDataUsed + spec.numPts, :] = spec.dIIc.T
            nDataUsed += spec.numPts
    if fitMag == 1:
        if magGeom.magGeomType != magGeomType:
            print('ERROR: miss-match in magGeomType flags!')
            import sys
            sys.exit()
        nDataUsed = 0
        nMagCoeff = 0
        if magGeom.magGeomType == 'full':
            nMagCoeff = 3 * 2 * magGeom.nTot
        elif magGeom.magGeomType == 'poloidal' or magGeom.magGeomType == 'pottor':
            nMagCoeff = 2 * 2 * magGeom.nTot
        elif magGeom.magGeomType == 'potential':
            nMagCoeff = 1 * 2 * magGeom.nTot
        allModeldV_lin = np.zeros((nDataTot, nMagCoeff))
        for spec in setSynSpec:
            nDataUsedNext = nDataUsed + spec.numPts
            if magGeom.magGeomType == 'full':
                for a in range(3):
                    allModeldV_lin[nDataUsed:nDataUsedNext, a * 2 * magGeom.nTot:(a * 2 + 1) * magGeom.nTot] = np.real(spec.dVIc[a, :, :]).T
                    allModeldV_lin[nDataUsed:nDataUsedNext, (a * 2 + 1) * magGeom.nTot:(a * 2 + 2) * magGeom.nTot] = np.imag(spec.dVIc[a, :, :]).T
            elif magGeom.magGeomType == 'poloidal' or magGeom.magGeomType == 'pottor':
                for a in range(2):
                    allModeldV_lin[nDataUsed:nDataUsedNext, a * 2 * magGeom.nTot:(a * 2 + 1) * magGeom.nTot] = np.real(spec.dVIc[a, :, :]).T
                    allModeldV_lin[nDataUsed:nDataUsedNext, (a * 2 + 1) * magGeom.nTot:(a * 2 + 2) * magGeom.nTot] = np.imag(spec.dVIc[a, :, :]).T
            elif magGeom.magGeomType == 'potential':
                allModeldV_lin[nDataUsed:nDataUsedNext, 0:magGeom.nTot] = np.real(spec.dVIc[0, :, :]).T
                allModeldV_lin[nDataUsed:nDataUsedNext, magGeom.nTot:2 * magGeom.nTot] = np.imag(spec.dVIc[0, :, :]).T
            nDataUsed += spec.numPts
    if (fitBri == 1) & (fitMag == 1):
        nDataUsed = 0
        allModeldVdBri = np.zeros([nDataTot, npBriMap])
        for spec in setSynSpec:
            allModeldVdBri[nDataUsed:nDataUsed + spec.numPts, :] = spec.dVdBri.T
            nDataUsed += spec.numPts
    if (fitBri == 1) & (fitMag == 1):
        allModeldIdV = np.zeros((nDataTot * 2, npBriMap + nMagCoeff))
        allModeldIdV[0:nDataTot, 0:npBriMap] = allModeldI
        allModeldIdV[nDataTot:nDataTot * 2, 0:npBriMap] = allModeldVdBri
        allModeldIdV[nDataTot:nDataTot * 2, npBriMap:npBriMap + nMagCoeff] = allModeldV_lin
    elif fitBri == 1:
        allModeldIdV = allModeldI
    elif fitMag == 1:
        allModeldIdV = allModeldV_lin
    else:
        print('ERROR: got no fittable parameters in packResponseMatrix()')
    return allModeldIdV

def packModelVector(setSynSpec, fitBri, fitMag):
    allModelI = np.empty(0)
    allModelV = np.empty(0)
    if fitBri == 1:
        for spec in setSynSpec:
            allModelI = np.append(allModelI, spec.IIc)
    if fitMag == 1:
        for spec in setSynSpec:
            allModelV = np.append(allModelV, spec.VIc)
    allModelIV = np.concatenate((allModelI, allModelV))
    return allModelIV

def packImageVector(briMap, magGeom, magGeomType, fitBri, fitMag):
    if magGeom.magGeomType != magGeomType:
        print('ERROR: miss-match in magGeomType flags!')
        import sys
        sys.exit()
    Image = np.empty(0)
    if fitBri == 1:
        Image = briMap.bright
    if fitMag == 1:
        if magGeomType == 'full':
            Image = np.concatenate([Image, np.real(magGeom.alpha), np.imag(magGeom.alpha), np.real(magGeom.beta), np.imag(magGeom.beta), np.real(magGeom.gamma), np.imag(magGeom.gamma)])
        elif magGeomType == 'poloidal':
            Image = np.concatenate([Image, np.real(magGeom.alpha), np.imag(magGeom.alpha), np.real(magGeom.beta), np.imag(magGeom.beta)])
        elif magGeomType == 'pottor':
            Image = np.concatenate([Image, np.real(magGeom.alpha), np.imag(magGeom.alpha), np.real(magGeom.gamma), np.imag(magGeom.gamma)])
        elif magGeomType == 'potential':
            Image = np.concatenate([Image, np.real(magGeom.alpha), np.imag(magGeom.alpha)])
    return Image

def unpackImageVector(Image, briMap, magGeom, magGeomType, fitBri, fitMag):
    if magGeom.magGeomType != magGeomType:
        print('ERROR: miss-match in second magGeomType flags!')
        import sys
        sys.exit()
    npBriMap = briMap.bright.shape[0]
    endBri = 0
    if fitBri == 1:
        briMap.bright[:] = Image[0:npBriMap]
        endBri = npBriMap
    if fitMag == 1:
        if magGeomType == 'full':
            magGeom.alpha = Image[endBri:endBri + magGeom.nTot] + Image[endBri + magGeom.nTot:endBri + magGeom.nTot * 2] * 1j
            magGeom.beta = Image[endBri + magGeom.nTot * 2:endBri + magGeom.nTot * 3] + Image[endBri + magGeom.nTot * 3:endBri + magGeom.nTot * 4] * 1j
            magGeom.gamma = Image[endBri + magGeom.nTot * 4:endBri + magGeom.nTot * 5] + Image[endBri + magGeom.nTot * 5:endBri + magGeom.nTot * 6] * 1j
        elif magGeomType == 'poloidal':
            magGeom.alpha = Image[endBri:endBri + magGeom.nTot] + Image[endBri + magGeom.nTot:endBri + magGeom.nTot * 2] * 1j
            magGeom.beta = Image[endBri + magGeom.nTot * 2:endBri + magGeom.nTot * 3] + Image[endBri + magGeom.nTot * 3:endBri + magGeom.nTot * 4] * 1j
            magGeom.gamma[:] = 0.0 + 0j
        elif magGeomType == 'pottor':
            magGeom.alpha = Image[endBri:endBri + magGeom.nTot] + Image[endBri + magGeom.nTot:endBri + magGeom.nTot * 2] * 1j
            magGeom.beta = Image[endBri:endBri + magGeom.nTot] + Image[endBri + magGeom.nTot:endBri + magGeom.nTot * 2] * 1j
            magGeom.gamma = Image[endBri + magGeom.nTot * 2:endBri + magGeom.nTot * 3] + Image[endBri + magGeom.nTot * 3:endBri + magGeom.nTot * 4] * 1j
        elif magGeomType == 'potential':
            magGeom.alpha = Image[endBri:endBri + magGeom.nTot] + Image[endBri + magGeom.nTot:endBri + magGeom.nTot * 2] * 1j
            magGeom.beta = Image[endBri:endBri + magGeom.nTot] + Image[endBri + magGeom.nTot:endBri + magGeom.nTot * 2] * 1j
            magGeom.gamma[:] = 0.0 + 0j
    return

class constantsMEM:

    def __init__(self, par, briMap, magGeom, nDataTot):
        self.npBriMap = 0
        self.npMagGeom = 0
        self.nDataTotIV = 0
        if par.fitBri == 1:
            self.npBriMap = briMap.bright.shape[0]
            self.nDataTotIV += nDataTot
        if par.fitMag == 1:
            if magGeom.magGeomType == 'full':
                self.npMagGeom = magGeom.nTot * 6
            elif magGeom.magGeomType == 'poloidal' or magGeom.magGeomType == 'pottor':
                self.npMagGeom = magGeom.nTot * 4
            elif magGeom.magGeomType == 'potential':
                self.npMagGeom = magGeom.nTot * 2
            else:
                self.npMagGeom = magGeom.nTot * 6
            self.nDataTotIV += nDataTot
        self.n1Model = self.npBriMap
        self.n2Model = self.n1Model
        if par.fEntropyBright == 2:
            self.n1Model = 0
            self.n2Model = self.npBriMap
        self.nModelTot = self.n2Model + self.npMagGeom

def setEntropyWeights(par, magGeom, sGrid):
    weightEntropy = 0
    if magGeom.magGeomType == 'poloidal':
        weightEntropyV = np.tile(magGeom.l, 4)
    if magGeom.magGeomType == 'pottor':
        weightEntropyV = np.tile(magGeom.l, 4)
    if magGeom.magGeomType == 'potential':
        weightEntropyV = np.tile(magGeom.l, 2)
    else:
        weightEntropyV = np.tile(magGeom.l, 6)
    weightEntropyI = sGrid.area * par.brightEntScale
    if par.fitBri == 1:
        weightEntropy = weightEntropyI
    if par.fitMag == 1:
        weightEntropy = weightEntropyV
    if (par.fitMag == 1) & (par.fitBri == 1):
        weightEntropy = np.concatenate((weightEntropyI, weightEntropyV))
    return weightEntropy
