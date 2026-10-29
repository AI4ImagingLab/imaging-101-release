"""Explicit selected ZDIpy mathematics, adapted from C. P. Folsom.
GPL-2.0; source revision 86d9f98c426a709f3298bab95fda5f6f5e65e111.
Only the bound full-vector, chi-square-target problem is supported.
"""
import numpy as np
import scipy.special
from scipy.special import factorial
from math import pi,sin,cos,acos
explicitConvolution=False

def legendre_values(degree, x):
    """Associated Legendre values and x derivatives, Condon--Shortley phase.

    SciPy 1.17 removed lpmn.  The derivative identity is
    (x*x-1) Pprime(l,m) = l*x*P(l,m)-(l+m)*P(l-1,m).
    The public midpoint mesh excludes the singular endpoints x=+/-1.
    """
    values=np.zeros((degree+1,degree+1)); deriv=np.zeros_like(values)
    for ell in range(degree+1):
        for order in range(ell+1):
            value=scipy.special.lpmv(order,ell,x)
            previous=scipy.special.lpmv(order,ell-1,x) if order<ell else 0.
            values[order,ell]=value
            deriv[order,ell]=(ell*x*value-(ell+order)*previous)/(x*x-1.)
    return values,deriv
class starGrid:

    def __init__(self, numPoints, period=0.0, mass=0.0, radiusEq=0.0, verbose=1):
        self.numPtsClat = numPoints
        self.numPtsLongEq = 2 * self.numPtsClat
        self.numPtsLong_Clat = np.zeros(self.numPtsClat, dtype=int)
        self.numPoints = 0
        for i in range(self.numPtsClat):
            _clat = pi * ((float(i) + 0.5) / float(self.numPtsClat))
            lnumPtsLong = int(round(sin(_clat) * float(self.numPtsLongEq)))
            self.numPtsLong_Clat[i] = lnumPtsLong
            self.numPoints += lnumPtsLong
        self.clat = np.zeros(self.numPoints)
        self.long = np.zeros(self.numPoints)
        self.radius = np.zeros(self.numPoints)
        self.dClat = np.zeros(self.numPoints)
        self.dLong = np.zeros(self.numPoints)
        self.dRadius = np.zeros(self.numPoints)
        self.gdark_local = None
        n = 0
        for i in range(self.numPtsClat):
            _clat = pi * ((float(i) + 0.5) / float(self.numPtsClat))
            _dClat = pi * (1.0 / float(self.numPtsClat))
            lnumPtsLong = int(self.numPtsLong_Clat[i])
            for j in range(lnumPtsLong):
                self.clat[n] = _clat
                self.dClat[n] = _dClat
                _long = 2.0 * pi * ((float(j) + 0.5) / float(lnumPtsLong))
                self.long[n] = _long
                _dLong = 2.0 * pi * (1.0 / float(lnumPtsLong))
                self.dLong[n] = _dLong
                n += 1
        self.radius[:] = 1.0
        self.dRadius[:] = 0.0
        self.fracOmegaCrit = 0.0
        if mass > 0.0 and radiusEq > 0.0:
            self.radius, self.fracOmegaCrit = self.calcRclatOblate(period, mass, radiusEq, verbose)
            self.dRadius = self.getOblateRadiusStep(period, mass, radiusEq)
        self.area = self.GetSurfaceArea()
        if verbose == 1:
            print('initiated a {:} point stellar grid  ( {:} clat, {:} long_equator)'.format(self.numPoints, self.numPtsClat, self.numPtsLongEq))

    def GetCartesianCells(self):
        x = self.radius * np.sin(self.clat) * np.cos(self.long)
        y = self.radius * np.sin(self.clat) * np.sin(self.long)
        z = self.radius * np.cos(self.clat)
        return np.array([x, y, z])

    def GetSurfaceArea(self):
        area = np.zeros(self.numPoints)
        if self.fracOmegaCrit > 0.0:

            def fThetaP(theta, wo):
                dS = 9.0 / wo ** 2 / np.sin(theta) * np.cos((np.pi + np.arccos(wo * np.sin(theta))) / 3.0) ** 2
                rs = 3.0 / (wo * np.sin(theta)) * np.cos((np.pi + np.arccos(wo * np.sin(theta))) / 3)
                gradPhi_r = 1.0 / rs ** 2 - 8.0 / 27.0 * rs * (wo * np.sin(theta)) ** 2
                gradPhi_theta = -8.0 / 27.0 * rs * wo ** 2 * np.sin(theta) * np.cos(theta)
                cosGamma = gradPhi_r / np.sqrt(gradPhi_r ** 2 + gradPhi_theta ** 2)
                thetaP = dS / cosGamma
                return thetaP
            from scipy.integrate import quad
            for i in range(self.numPoints):
                if self.clat[i] != self.clat[i - 1]:
                    intTheta, errTheta = quad(fThetaP, self.clat[i] - 0.5 * self.dClat[i], self.clat[i] + 0.5 * self.dClat[i], args=self.fracOmegaCrit)
                area[i] = self.dLong[i] * intTheta
        else:
            area = self.dLong * (np.cos(self.clat - 0.5 * self.dClat) - np.cos(self.clat + 0.5 * self.dClat))
        return area

    def GetCartesianRotVel(self):
        velX = np.negative(self.radius * np.sin(self.clat) * np.sin(self.long))
        velY = self.radius * np.sin(self.clat) * np.cos(self.long)
        velZ = np.zeros(len(self.clat))
        velVec = np.array([velX, velY, velZ])
        iEquator = np.argmin(np.abs(self.clat - 0.5 * np.pi))
        velVec = velVec / self.radius[iEquator]
        return velVec

    def getCartesianNormals(self):
        if self.fracOmegaCrit <= 0.0:
            vNormals = self.GetCartesianCells()
            lenThisPoint = np.sqrt(vNormals[0, :] ** 2 + vNormals[1, :] ** 2 + vNormals[2, :] ** 2)
            vNormals /= lenThisPoint
        else:
            wo = self.fracOmegaCrit
            sr = self.radius
            sclat = self.clat
            slong = self.long
            gradPot = np.array([-8.0 / 27.0 * wo ** 2 * sr * np.sin(sclat) ** 2 + 1.0 / sr ** 2, -8.0 / 27.0 * wo ** 2 * sr * np.sin(sclat) * np.cos(sclat), np.zeros_like(sclat)])
            gpotnorm = gradPot / np.sqrt(gradPot[0] ** 2 + gradPot[1] ** 2)
            R = np.array([[np.sin(sclat) * np.cos(slong), np.cos(sclat) * np.cos(slong), -np.sin(slong)], [np.sin(sclat) * np.sin(slong), np.cos(sclat) * np.sin(slong), np.cos(slong)], [np.cos(sclat), -np.sin(sclat), np.zeros_like(sclat)]])
            vNormals = np.einsum('ijk,jk->ik', R, gpotnorm)
        return vNormals

    def calcRclatOblate(self, period, mass, radius, verbose=1):
        G = 6.67408e-11
        Msun = 1.98892e+30
        Rsun = 695500000.0
        Omega = 2 * np.pi / (period * 86400.0)
        Omega_break = np.sqrt(8.0 / 27.0 * G * (mass * Msun) / (radius * Rsun) ** 3)
        wo = Omega / Omega_break
        if wo > 1.0:
            print(('Error: star above breakup velocity (Omega/Omega_break {:}), \n' + '   for given mass {:} Msun, radius {:} Rsun, and period {:} d').format(wo, mass, radius, period))
            print('Assuming breakup velocity for geometry calculations')
            wo = 1.0
        elif wo < 1e-08:
            wo = 1e-08
        obl = 3.0 / wo * cos((np.pi + acos(wo)) / 3.0)
        if verbose:
            print('The Oblateness of the star: Req/Rp = {:} ({:} of breakup)'.format(obl, wo))
        x_clat = 3.0 / (wo * np.sin(self.clat)) * np.cos((np.pi + np.arccos(wo * np.sin(self.clat))) / 3)
        return (x_clat, wo)

    def getOblateRadiusStep(self, period, mass, radiusEq):
        if mass > 0.0 and radiusEq > 0.0:
            _refClat = self.clat
            self.clat = _refClat + 0.499999999 * self.dClat
            radiusP, tmp = self.calcRclatOblate(period, mass, radiusEq, verbose=False)
            self.clat = _refClat - 0.499999999 * self.dClat
            radiusM, tmp = self.calcRclatOblate(period, mass, radiusEq, verbose=False)
            self.clat = _refClat
            dRadius = radiusP - radiusM
        else:
            dRadius = np.zeros_like(self.clat)
        return dRadius

    def gravityDarkening(self, gravDarkCoeff):
        bCalc = True
        if self.gdark_local is not None:
            if self.gravDarkCoeff == gravDarkCoeff:
                gdark_local = self.gdark_local
                bCalc = False
        if bCalc:
            wo = self.fracOmegaCrit
            if wo > 0.0 and gravDarkCoeff > 0.0:
                gr = -1.0 / self.radius ** 2 + 8.0 / 27.0 * self.radius * (wo * np.sin(self.clat)) ** 2
                gtheta = 8.0 / 27.0 * self.radius * np.sin(self.clat) * np.cos(self.clat) * wo ** 2
                gdark_local = np.sqrt(gr ** 2 + gtheta ** 2) ** gravDarkCoeff
            else:
                gdark_local = np.ones_like(self.clat)
            self.gravDarkCoeff = gravDarkCoeff
            self.gdark_local = gdark_local
        return gdark_local

class visibleGrid:

    def __init__(self, starGrid, inclination, cycleAtClat, period, dOmega):
        self.starGrid = starGrid
        viewClat = inclination
        viewLong = -cycleAtClat * 2.0 * np.pi
        viewX = 1.0 * np.sin(viewClat) * np.cos(viewLong)
        viewY = 1.0 * np.sin(viewClat) * np.sin(viewLong)
        viewZ = 1.0 * np.cos(viewClat) * np.ones(viewLong.shape)
        vView = np.array([viewX, viewY, viewZ])
        lenView = np.linalg.norm(vView, axis=0)
        self.vViewCart = vView / lenView
        vNormals = starGrid.getCartesianNormals()
        dotprod = np.einsum('ij,ij->j', vNormals, vView)
        _cosViewAngle = dotprod / lenView
        self.viewAngle = np.arccos(_cosViewAngle)
        self.visible = np.zeros(starGrid.numPoints, dtype=int)
        for i in range(starGrid.numPoints):
            if (self.viewAngle[i] > 0.0) & (self.viewAngle[i] < pi * 0.5):
                self.visible[i] = 1
            else:
                self.visible[i] = 0
        self.projArea = starGrid.area * _cosViewAngle
        vVel = starGrid.GetCartesianRotVel()
        velDiffRot = calcVelDiffrotFactor(period, dOmega, starGrid.clat)
        vVel = vVel * velDiffRot
        self.velRotProj = np.einsum('ij,ij->j', -self.vViewCart, vVel)
        self.gravityDarkening = self.starGrid.gravityDarkening

def calcOmegaClat(period, dOmega, clat):
    omega_eq = 2.0 * np.pi / period
    omega_clat = omega_eq - dOmega * np.cos(clat) ** 2
    return omega_clat

def getCyclesClat(period, dOmega, jDates, jDateRef, clat):
    omega_clat = calcOmegaClat(period, dOmega, clat)
    period_clat = 2.0 * np.pi / omega_clat
    cycleAtClat = (jDates[:, np.newaxis] - jDateRef) / period_clat[np.newaxis, :]
    return cycleAtClat

def calcVelDiffrotFactor(period, dOmega, clat):
    omega_eq = 2.0 * np.pi / period
    omega_clat = calcOmegaClat(period, dOmega, clat)
    norm_clat = omega_clat / omega_eq
    return norm_clat
class magSphHarmoics:

    def __init__(self, nHarmics):
        self.nl = nHarmics
        self.nTot = self.nl * (self.nl + 1) // 2 + self.nl
        self.l = np.zeros(self.nTot)
        self.m = np.zeros(self.nTot)
        index = 0
        for i in range(self.nl):
            for j in range(i + 2):
                self.l[index] = i + 1
                self.m[index] = j
                index += 1
        self.magGeomType = 'full'
        self.alpha = np.zeros(self.nTot, dtype=complex)
        self.beta = np.zeros(self.nTot, dtype=complex)
        self.gamma = np.zeros(self.nTot, dtype=complex)
        self.clat = []
        self.lon = []
        self.XTerm = []
        self.YTerm = []
        self.ZTerm = []

    def initMagGeom(self, clat, lon):
        nCells = len(clat)
        cosClat = np.cos(clat)
        self.clat = clat
        self.lon = lon
        pLegendre_all = np.zeros((self.nTot, nCells))
        dPdTh_all = np.zeros((self.nTot, nCells))
        indTri = np.tril_indices(self.nl + 1)
        indTri = (indTri[1][1:], indTri[0][1:])
        for i in range(nCells):
            if cosClat[i] == cosClat[i - 1]:
                pLegendre_all[..., i] = pLegendre_all[..., i - 1]
                dPdTh_all[..., i] = dPdTh_all[..., i - 1]
            else:
                pLegendre = legendre_values(self.nl, cosClat[i])
                pLegendre_all[..., i] = pLegendre[0][indTri]
                dPdTh_all[..., i] = pLegendre[1][indTri]
        dPdTh_all = dPdTh_all * -np.sqrt(1.0 - cosClat ** 2)
        expTerm_all = np.exp(1j * np.outer(self.m, lon))
        cTerm_lm = np.sqrt((2.0 * self.l + 1.0) / (4.0 * np.pi) * factorial(self.l - self.m) / factorial(self.l + self.m))
        c2Term_lm = cTerm_lm / (self.l + 1.0)
        c3Term_lm = 1j * c2Term_lm * self.m
        invSinClat = 1.0 / np.sin(clat)
        self.YTerm = cTerm_lm[:, np.newaxis] * pLegendre_all * expTerm_all
        self.XTerm = c3Term_lm[:, np.newaxis] * invSinClat[np.newaxis, :] * pLegendre_all * expTerm_all
        self.ZTerm = c2Term_lm[:, np.newaxis] * dPdTh_all * expTerm_all

    def getAllMagVectors(self):
        Br = np.real(np.dot(self.alpha, self.YTerm))
        Bclat = -np.real(np.dot(self.beta, self.ZTerm) + np.dot(self.gamma, self.XTerm))
        Blon = -np.real(np.dot(self.beta, self.XTerm) - np.dot(self.gamma, self.ZTerm))
        vMagAll = np.array([Br, Bclat, Blon])
        return vMagAll

    def getAllMagDerivs(self):
        dBr_dAlpha = np.conj(self.YTerm)
        if self.magGeomType == 'full':
            dBclat_dBeta = -np.conj(self.ZTerm)
            dBclat_dGamma = -np.conj(self.XTerm)
            dBlon_dBeta = -np.conj(self.XTerm)
            dBlon_dGamma = np.conj(self.ZTerm)
            derivMagAll = np.concatenate((dBr_dAlpha, dBclat_dBeta, dBclat_dGamma, dBlon_dBeta, dBlon_dGamma), axis=0)
            derivMagAll = derivMagAll.reshape((5, self.nTot, self.YTerm.shape[1]))
        elif self.magGeomType == 'poloidal':
            dBclat_dBeta = -np.conj(self.ZTerm)
            dBlon_dBeta = -np.conj(self.XTerm)
            derivMagAll = np.concatenate((dBr_dAlpha, dBclat_dBeta, dBlon_dBeta), axis=0)
            derivMagAll = derivMagAll.reshape((3, self.nTot, self.YTerm.shape[1]))
        elif self.magGeomType == 'pottor':
            dBclat_dAlpha = -np.conj(self.ZTerm)
            dBclat_dGamma = -np.conj(self.XTerm)
            dBlon_dAlpha = -np.conj(self.XTerm)
            dBlon_dGamma = np.conj(self.ZTerm)
            derivMagAll = np.concatenate((dBr_dAlpha, dBclat_dAlpha, dBclat_dGamma, dBlon_dAlpha, dBlon_dGamma), axis=0)
            derivMagAll = derivMagAll.reshape((5, self.nTot, self.YTerm.shape[1]))
        elif self.magGeomType == 'potential':
            dBclat_dAlpha = -np.conj(self.ZTerm)
            dBlon_dAlpha = -np.conj(self.XTerm)
            derivMagAll = np.concatenate((dBr_dAlpha, dBclat_dAlpha, dBlon_dAlpha))
            derivMagAll = derivMagAll.reshape((3, self.nTot, self.YTerm.shape[1]))
        return derivMagAll

    def getAllMagVectorsCart(self):
        sinClat = np.sin(self.clat)
        cosClat = np.cos(self.clat)
        sinLon = np.sin(self.lon)
        cosLon = np.cos(self.lon)
        vecB = self.getAllMagVectors()
        Br = vecB[0, :]
        Bclat = vecB[1, :]
        Blon = vecB[2, :]
        Bx = Br * sinClat * cosLon + Bclat * cosClat * cosLon - Blon * sinLon
        By = Br * sinClat * sinLon + Bclat * cosClat * sinLon + Blon * cosLon
        Bz = Br * cosClat - Bclat * sinClat
        Bcart = np.concatenate((Bx[np.newaxis], By[np.newaxis], Bz[np.newaxis]), axis=0)
        return Bcart

    def getAllMagDerivsCart(self):
        sinClat = np.sin(self.clat)
        cosClat = np.cos(self.clat)
        sinLon = np.sin(self.lon)
        cosLon = np.cos(self.lon)
        dB = self.getAllMagDerivs()
        dBr_dAlpha = dB[0, ...]
        if self.magGeomType == 'full':
            dBclat_dBeta = dB[1, ...]
            dBclat_dGamma = dB[2, ...]
            dBlon_dBeta = dB[3, ...]
            dBlon_dGamma = dB[4, ...]
        elif self.magGeomType == 'poloidal':
            dBclat_dBeta = dB[1, ...]
            dBlon_dBeta = dB[2, ...]
        elif self.magGeomType == 'pottor':
            dBclat_dAlpha = dB[1, ...]
            dBclat_dGamma = dB[2, ...]
            dBlon_dAlpha = dB[3, ...]
            dBlon_dGamma = dB[4, ...]
        elif self.magGeomType == 'potential':
            dBclat_dAlpha = dB[1, ...]
            dBlon_dAlpha = dB[2, ...]
        if self.magGeomType == 'full':
            dBx_dAlpha = dBr_dAlpha * (sinClat * cosLon)[np.newaxis, :]
            dBy_dAlpha = dBr_dAlpha * (sinClat * sinLon)[np.newaxis, :]
            dBz_dAlpha = dBr_dAlpha * cosClat[np.newaxis, :]
            dBx_dBeta = dBclat_dBeta * (cosClat * cosLon)[np.newaxis, :] - dBlon_dBeta * sinLon[np.newaxis, :]
            dBy_dBeta = dBclat_dBeta * (cosClat * sinLon)[np.newaxis, :] + dBlon_dBeta * cosLon[np.newaxis, :]
            dBz_dBeta = -dBclat_dBeta * sinClat[np.newaxis, :]
            dBx_dGamma = dBclat_dGamma * (cosClat * cosLon)[np.newaxis, :] - dBlon_dGamma * sinLon[np.newaxis, :]
            dBy_dGamma = dBclat_dGamma * (cosClat * sinLon)[np.newaxis, :] + dBlon_dGamma * cosLon[np.newaxis, :]
            dBz_dGamma = -dBclat_dGamma * sinClat[np.newaxis, :]
        elif self.magGeomType == 'poloidal':
            dBx_dAlpha = dBr_dAlpha * (sinClat * cosLon)[np.newaxis, :]
            dBy_dAlpha = dBr_dAlpha * (sinClat * sinLon)[np.newaxis, :]
            dBz_dAlpha = dBr_dAlpha * cosClat[np.newaxis, :]
            dBx_dBeta = dBclat_dBeta * (cosClat * cosLon)[np.newaxis, :] - dBlon_dBeta * sinLon[np.newaxis, :]
            dBy_dBeta = dBclat_dBeta * (cosClat * sinLon)[np.newaxis, :] + dBlon_dBeta * cosLon[np.newaxis, :]
            dBz_dBeta = -dBclat_dBeta * sinClat[np.newaxis, :]
        elif self.magGeomType == 'pottor':
            dBx_dAlpha = dBr_dAlpha * (sinClat * cosLon)[np.newaxis, :] + dBclat_dAlpha * (cosClat * cosLon)[np.newaxis, :] - dBlon_dAlpha * sinLon[np.newaxis, :]
            dBy_dAlpha = dBr_dAlpha * (sinClat * sinLon)[np.newaxis, :] + dBclat_dAlpha * (cosClat * sinLon)[np.newaxis, :] + dBlon_dAlpha * cosLon[np.newaxis, :]
            dBz_dAlpha = dBr_dAlpha * cosClat[np.newaxis, :] - dBclat_dAlpha * sinClat[np.newaxis, :]
            dBx_dGamma = dBclat_dGamma * (cosClat * cosLon)[np.newaxis, :] - dBlon_dGamma * sinLon[np.newaxis, :]
            dBy_dGamma = dBclat_dGamma * (cosClat * sinLon)[np.newaxis, :] + dBlon_dGamma * cosLon[np.newaxis, :]
            dBz_dGamma = -dBclat_dGamma * sinClat[np.newaxis, :]
        elif self.magGeomType == 'potential':
            dBx_dAlpha = dBr_dAlpha * (sinClat * cosLon)[np.newaxis, :] + dBclat_dAlpha * (cosClat * cosLon)[np.newaxis, :] - dBlon_dAlpha * sinLon[np.newaxis, :]
            dBy_dAlpha = dBr_dAlpha * (sinClat * sinLon)[np.newaxis, :] + dBclat_dAlpha * (cosClat * sinLon)[np.newaxis, :] + dBlon_dAlpha * cosLon[np.newaxis, :]
            dBz_dAlpha = dBr_dAlpha * cosClat[np.newaxis, :] - dBclat_dAlpha * sinClat[np.newaxis, :]
        if self.magGeomType == 'full':
            dBcart = np.concatenate((dBx_dAlpha, dBx_dBeta, dBx_dGamma, dBy_dAlpha, dBy_dBeta, dBy_dGamma, dBz_dAlpha, dBz_dBeta, dBz_dGamma))
            dBcart = np.reshape(dBcart, (3, 3, dB.shape[1], dB.shape[2]))
        elif self.magGeomType == 'poloidal':
            dBcart = np.concatenate((dBx_dAlpha, dBx_dBeta, dBy_dAlpha, dBy_dBeta, dBz_dAlpha, dBz_dBeta))
            dBcart = np.reshape(dBcart, (3, 2, dB.shape[1], dB.shape[2]))
        elif self.magGeomType == 'pottor':
            dBcart = np.concatenate((dBx_dAlpha, dBx_dGamma, dBy_dAlpha, dBy_dGamma, dBz_dAlpha, dBz_dGamma))
            dBcart = np.reshape(dBcart, (3, 2, dB.shape[1], dB.shape[2]))
        elif self.magGeomType == 'potential':
            dBcart = np.concatenate((dBx_dAlpha, dBy_dAlpha, dBz_dAlpha))
            dBcart = np.reshape(dBcart, (3, 1, dB.shape[1], dB.shape[2]))
        return dBcart
def limbDarkening(coeffEta, angle):
    limbD = 1.0 - coeffEta + coeffEta * np.cos(angle)
    return limbD

class localProfileAndDeriv:

    def __init__(self, lineData, numPts, wlGrid):
        c = 299792.458
        self.wl = wlGrid
        self.Q = np.zeros(numPts)
        self.U = np.zeros(numPts)
        if explicitConvolution == False and lineData.instRes > 0.0:
            velInstRes = c / lineData.instRes * 0.6005612043932249
            widthGauss = np.sqrt(lineData.widthGauss[0] ** 2 + velInstRes ** 2)
            widthLorentz = lineData.widthLorentz[0] * (lineData.widthGauss[0] / widthGauss)
            lineStr = lineData.str[0] * (lineData.widthGauss[0] / widthGauss)
        elif explicitConvolution == False and lineData.instRes < 0.0:
            print('Warning: convolution with an instrumental profile may not be performed')
            widthGauss = lineData.widthGauss[0]
            widthLorentz = lineData.widthLorentz[0]
            lineStr = lineData.str[0]
        else:
            widthGauss = lineData.widthGauss[0]
            widthLorentz = lineData.widthLorentz[0]
            lineStr = lineData.str[0]
        w4 = np.zeros(self.wl.shape, dtype=complex)
        widthGaussWl = widthGauss / c * lineData.wl0[0]
        wlGaussNorm = (lineData.wl0[0] - self.wl) / widthGaussWl
        z = widthLorentz - 1j * wlGaussNorm
        zz = z * z
        s = np.abs(wlGaussNorm) + widthLorentz
        con1 = np.where(s >= 15.0)
        zt = z[con1]
        w4[con1] = 0.56418958355 * zt / (0.5 + zt * zt)
        con2 = np.where((s >= 5.5) & (s < 15.0))
        zt = z[con2]
        zzt = zz[con2]
        w4[con2] = zt * (1.4104739589 + 0.56418958355 * zzt) / ((3.0 + zzt) * zzt + 0.7499999999)
        con3 = np.where((widthLorentz >= 0.195 * np.abs(wlGaussNorm) - 0.176) & (s < 5.5))
        zt = z[con3]
        w4[con3] = (16.4954955 + zt * (20.2093334 + zt * (11.9648172 + zt * (3.77898687 + zt * 0.564223565)))) / (16.4954955 + zt * (38.8236274 + zt * (39.2712051 + zt * (21.692737 + zt * (6.69939801 + zt)))))
        con4 = np.where(w4 == 0.0 + 0j)
        zt = z[con4]
        zzt = zz[con4]
        w4[con4] = np.exp(zzt) - zt * (36183.30536 - zzt * (3321.990492 - zzt * (1540.786893 - zzt * (219.0312964 - zzt * (35.7668278 - zzt * (1.320521697 - zzt * 0.5641900381)))))) / (32066.59372 - zzt * (24322.84021 - zzt * (9022.227659 - zzt * (2186.181081 - zzt * (364.2190727 - zzt * (61.57036588 - zzt * (1.841438936 - zzt)))))))
        dVoigtdWl = lineStr / widthGaussWl * 2.0 * (-wlGaussNorm * w4.real + widthLorentz * w4.imag)
        self.Iunscaled = 1.0 - lineStr * w4.real
        gCoeff = -4.6686e-12 * lineData.wl0[0] ** 2 * lineData.g[0]
        self.dIdWlG = gCoeff * dVoigtdWl

    def updateProfDeriv(self, lineData, Blos, dBlos_d, brightMap, numPts, wlGrid, surfaceScale, calcDI, calcDV):
        contin = surfaceScale * brightMap.bright
        self.I = contin[np.newaxis, :] * self.Iunscaled
        self.V = (Blos * contin)[np.newaxis, :] * self.dIdWlG
        self.Ic = np.sum(contin)
        invSumIc0 = 1.0 / np.sum(self.Ic)
        self.I = np.sum(self.I, axis=1) * invSumIc0
        self.V = np.sum(self.V, axis=1) * invSumIc0
        if calcDV == 1:
            dIdWlGcont = self.dIdWlG * contin[np.newaxis, :]
            self.dVsum = np.einsum('ijl,kl->ikj', dBlos_d, dIdWlGcont)
            self.dVsum = np.swapaxes(self.dVsum, 1, 2)
            self.dVsum *= invSumIc0
        else:
            self.dVsum = 0
        if calcDI == 1:
            self.dIcsum = (surfaceScale * invSumIc0)[:, np.newaxis] * (self.Iunscaled.T - self.I[np.newaxis, :])
        else:
            self.dIcsum = 0
        if (calcDI == 1) & (calcDV == 1):
            self.dVdBright = (surfaceScale * invSumIc0)[:, np.newaxis] * ((Blos[np.newaxis, :] * self.dIdWlG).T - self.V[np.newaxis, :])
        else:
            self.dVdBright = 0

class diskIntProfAndDeriv:

    def __init__(self, visibleGrid, vMagCart, dMagCart, brightMap, lData, velEq, wlGrid, calcDI, calcDV):
        c = 299792.458
        self.wl = wlGrid
        self.numPts = wlGrid.shape[0]
        self.wlStart = wlGrid[0]
        self.wlEnd = wlGrid[-1]
        self.IIc = np.zeros(self.numPts)
        self.QIc = np.zeros(self.numPts)
        self.UIc = np.zeros(self.numPts)
        self.VIc = np.zeros(self.numPts)
        vrCell = velEq * visibleGrid.velRotProj
        self.wlCells = np.outer(self.wl, 1.0 / (1.0 + vrCell / c))
        self.prof = localProfileAndDeriv(lData, self.numPts, self.wlCells)
        self.updateIntProfDeriv(visibleGrid, vMagCart, dMagCart, brightMap, lData, calcDI, calcDV)

    def updateIntProfDeriv(self, visibleGrid, vMagCart, dMagCart, brightMap, lData, calcDI, calcDV):
        self.calcDI = calcDI
        self.calcDV = calcDV
        vView = visibleGrid.vViewCart
        projBlos = np.einsum('ij,ij->j', vView, vMagCart)
        if calcDV == 1:
            dBlos_d = np.einsum('il,ijkl->jkl', vView, dMagCart)
        else:
            dBlos_d = 0
        limbDark = limbDarkening(lData.limbDark, visibleGrid.viewAngle)
        gDark = visibleGrid.gravityDarkening(lData.gravDark)
        surfaceScale = limbDark * gDark * visibleGrid.projArea * visibleGrid.visible
        self.prof.updateProfDeriv(lData, projBlos, dBlos_d, brightMap, self.numPts, self.wlCells, surfaceScale, calcDI, calcDV)
        self.IIc = self.prof.I
        self.VIc = self.prof.V
        self.dVIc = self.prof.dVsum
        self.dIIc = self.prof.dIcsum
        self.dVdBri = self.prof.dVdBright

    def convolveIGnumpy(self, fwhm):
        """Instrumental Gaussian broadening is already in the local Voigt width."""
        return

def getAllProfDiriv(par, listGridView, vecMagCart, dMagCart0, briMap, lineData, wlSynSet):
    nObs = 0
    setSynSpec = []
    for phase in par.cycleList:
        spec = diskIntProfAndDeriv(listGridView[nObs], vecMagCart, dMagCart0, briMap, lineData, par.velEq, wlSynSet[nObs], 0, par.calcDV)
        spec.convolveIGnumpy(par.instrumentRes)
        setSynSpec += [spec]
        nObs += 1
    return setSynSpec

def calcSynEW(spec):
    equivWidApprox = 0.0
    for i in range(spec.wl.shape[0] - 1):
        equivWidApprox += (1.0 - spec.IIc[i]) * (spec.wl[i + 1] - spec.wl[i])
    equivWidApprox += (1.0 - spec.IIc[-1]) * (spec.wl[-1] - spec.wl[-2])
    return equivWidApprox
