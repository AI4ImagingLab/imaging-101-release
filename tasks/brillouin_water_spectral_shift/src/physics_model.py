"""Explicit Lorentzian, circular-trace and nonlinear VIPA mathematics.
Adapted numerical equations from bmlab developers, GPL-3.0; see LICENSE.
No upstream package is used by this module.
"""
from types import SimpleNamespace
import numpy as np
from scipy.optimize import least_squares,minimize,fmin
from scipy.signal import find_peaks
class FitError(Exception):pass

def setup():
    """Instrument geometry and independent water/methanol calibration standards."""
    c=299792458.;wavelength=780.24e-9;n=1.45367;d=.006743;theta=.8*2*np.pi/360;f=.2
    fsr=c/(2*n*d*np.cos(theta));order=round(c/(wavelength*fsr))
    vipa=SimpleNamespace(m=order,order=0,FSR=fsr)
    pars=(2*np.pi*n*d*np.cos(theta)/c,-2*np.pi*n*d*np.tan(theta)/(c*f)*np.sqrt(1-(n*np.sin(theta))**2),-np.pi/c*d*np.cos(theta)/f**2)
    calibration=SimpleNamespace(num_brillouin_samples=2,shifts=np.array([0,3.78e9,5.066e9,-5.066e9,-3.78e9,0]),orders=np.array([0,0,0,1,1,1]))
    return SimpleNamespace(pixel_size=6.5e-6,VIPA_PARAMS=pars,vipa=vipa,calibration=calibration,f0=c/wavelength)

def lorentz(x, w0, fwhm, intensity):
    with np.errstate(divide='ignore', invalid='ignore'):
        return intensity * (fwhm / 2) ** 2 / ((x - w0) ** 2 + (fwhm / 2) ** 2)

def fit_lorentz(x, y):
    w0_guess = float(x[np.argmax(y)])
    offset_guess = (y[0] + y[-1]) / 2.0
    intensity_guess = np.max(y) - offset_guess
    fwhm_guess = float(2 * np.abs(w0_guess - x[np.argmax(y > offset_guess + intensity_guess / 2)]))
    if fwhm_guess <= 0.0:
        fwhm_guess = 2 * (x[-1] - x[0]) / x.shape[0]

    def error(params, xdata, ydata):
        return (ydata - lorentz(xdata, *params[0:3]) - params[3]) ** 2
    bounds_lower = (x[0], (x[-1] - x[0]) / x.shape[0], 0, 0)
    bounds_upper = (x[-1], x[-1] - x[0], 2 * np.max(y), 2 * np.max(y))
    bounds = (bounds_lower, bounds_upper)
    opt_result = least_squares(error, x0=(w0_guess, fwhm_guess, intensity_guess, offset_guess), args=(x, y), bounds=bounds)
    if not opt_result.success:
        raise FitError('Lorentz fit failed.')
    w0, fwhm, intensity, offset = opt_result.x
    return (w0, fwhm, intensity, offset)

def fit_double_lorentz(x, y, bounds_w0=None, bounds_fwhm=None):
    offset_guess = (y[0] + y[-1]) / 2.0
    fwhm_guess = 10 * (x[-1] - x[0]) / x.shape[0] * np.ones(2)
    intensity_guess = np.max(y) - offset_guess
    peaks, properties = find_peaks(y, prominence=1)
    idx = np.argsort(properties['prominences'])[::-1]
    if len(idx) < 2:
        return
    idx_sort = np.sort(peaks[idx[0:2]])
    w0_guess = list(x[idx_sort])

    def error(params, xdata, ydata):
        return (ydata - lorentz(xdata, *params[0:3]) - lorentz(xdata, *params[3:6]) - params[6]) ** 2
    if bounds_w0 is None and bounds_fwhm is None:
        bounds = (-np.inf, np.inf)
    else:
        bounds_lower = -np.inf * np.ones(7)
        bounds_upper = np.inf * np.ones(7)
        fwhm_lower_bound = (x[-1] - x[0]) / x.shape[0]
        bounds_lower[1] = fwhm_lower_bound
        bounds_lower[2] = 0
        bounds_lower[4] = fwhm_lower_bound
        bounds_lower[5] = 0
        bounds_lower[6] = 0
        if bounds_w0 is not None:
            bounds_lower[0] = bounds_w0[0][0]
            bounds_upper[0] = bounds_w0[0][1]
            bounds_lower[3] = bounds_w0[1][0]
            bounds_upper[3] = bounds_w0[1][1]
            bounds_w0_center = [np.mean(np.clip(bound, *x[::len(x) - 1])) for bound in bounds_w0]
            w0_guess.sort(reverse=bool(bounds_w0_center[0] > bounds_w0_center[1]))
            w0_guess = [np.clip(guess, *bounds_w0[idx]) for idx, guess in enumerate(w0_guess)]
        if bounds_fwhm is not None:
            bounds_lower[1] = bounds_fwhm[0][0]
            bounds_upper[1] = bounds_fwhm[0][1]
            bounds_lower[4] = bounds_fwhm[1][0]
            bounds_upper[4] = bounds_fwhm[1][1]
            fwhm_guess = [np.clip(guess, *bounds_fwhm[idx]) for idx, guess in enumerate(fwhm_guess)]
        bounds = (bounds_lower, bounds_upper)
    opt_result = least_squares(error, x0=(w0_guess[0], fwhm_guess[0], intensity_guess, w0_guess[1], fwhm_guess[1], intensity_guess, offset_guess), args=(x, y), bounds=bounds)
    if not opt_result.success:
        raise FitError('Lorentz fit failed.')
    res = opt_result.x
    w0s, fwhms, intens = ((res[0], res[3]), (res[1], res[4]), (res[2], res[5]))
    offset = res[6]
    return (w0s, fwhms, intens, offset)

def fit_circle(points):
    """
    Fits a circle to a given set of points. Returnes the center and the radius
    of the cricle. The inital parameters for the fiting process are arbitrarily
    chosen.

    Parameters
    ----------
    points: list of tuples

    Returns
    -------
    center_opt: tuple
                Coordinates of the circle center (x,y)
    radius_opt: float
                Radius of the circle
    """
    center, radius = calculate_exact_circle(points)
    if len(points) <= 3:
        return (center, radius)
    param_guess = [center[0], center[1], radius]
    x_coords = np.array([xy[0] for xy in points])
    y_coords = np.array([xy[1] for xy in points])
    bnds = ((None, None), (None, None), (0.0, None))
    opt_result = minimize(_circle_opt, param_guess, args=(x_coords, y_coords), bounds=bnds)
    return ((opt_result['x'][0], opt_result['x'][1]), abs(opt_result['x'][2]))

def _circle_opt(c, x_coord, y_coord):
    """
    Cost function to fit a circle to a given set of points.

    Parameters
    ----------
    c: array like
        Parameters to optimize: [x coordinates of circle center,
        y coordinates of circle center, radius of the cricle]
    x_coord:
        x coordinates of the points to fit
    y_coord
        y coordinates of the points to fit

    Returns
    -------
    out: float
        Cost value representing the deviation of the given set of points to
        the circle, which is represented by the parameter vector c.
    """
    return np.sum(((x_coord - c[0]) ** 2 + (y_coord - c[1]) ** 2 - c[2] ** 2) ** 2)

def calculate_exact_circle(points):
    if len(points) < 3:
        return
    points = [points[idx] for idx in np.linspace(0, len(points) - 1, 3, dtype=int)]
    if are_points_on_line(points):
        x, y = points[0]
        points[0] = (x - 1, y)
    x1 = points[0][0]
    y1 = points[0][1]
    x2 = points[1][0]
    y2 = points[1][1]
    x3 = points[2][0]
    y3 = points[2][1]
    x12 = x1 - x2
    x13 = x1 - x3
    y12 = y1 - y2
    y13 = y1 - y3
    sx13 = x1 ** 2 - x3 ** 2
    sy13 = y1 ** 2 - y3 ** 2
    sx21 = x2 ** 2 - x1 ** 2
    sy21 = y2 ** 2 - y1 ** 2
    x0 = -(sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) / (2 * (x12 * y13 - x13 * y12))
    y0 = -(sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) / (2 * (y12 * x13 - y13 * x12))
    c = -x1 ** 2 - y1 ** 2 + 2 * x0 * x1 + 2 * y0 * y1
    r = np.sqrt(x0 ** 2 + y0 ** 2 - c)
    return ((x0, y0), r)

def are_points_on_line(points):
    """
    This function checks if the first three points in the given
    list lay on the same line
    Parameters
    ----------
    points: list of tuples representing points

    Returns
    -------
    boolean
    Whether the given points lay on the same line
    """
    if len(points) < 3:
        return True
    m = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0])
    n = points[0][1] - m * points[0][0]
    for point in points[:3]:
        if point[1] != m * point[0] + n:
            return False
    return True

def fit_vipa(peaks, setup):
    """
    Fits the VIPA frequency axis

    Parameters
    ----------
    peaks: The peak positions
    setup: The setup parameters

    Returns
    -------
    The fitted VIPA parameters
    """
    if peaks is None:
        return
    if len(peaks) < setup.calibration.num_brillouin_samples * 2 + 2:
        return
    r0 = peaks[0] * setup.pixel_size
    v1 = (setup.VIPA_PARAMS[0] + setup.VIPA_PARAMS[1] * r0 + setup.VIPA_PARAMS[2] * r0 ** 2) / ((setup.vipa.m + setup.vipa.order) * np.pi)
    v2 = (setup.VIPA_PARAMS[1] + setup.VIPA_PARAMS[2] * 2 * r0) / ((setup.vipa.m + setup.vipa.order) * np.pi) * setup.pixel_size
    v3 = setup.VIPA_PARAMS[2] / ((setup.vipa.m + setup.vipa.order) * np.pi) * setup.pixel_size ** 2
    vipa_start = np.array([v1, v2, v3, setup.vipa.FSR])

    def error(vipa_params, peaks1, shifts1):
        fsr = vipa_params[3]
        frequencies = VIPA(peaks1, vipa_params) - setup.f0
        d1 = frequencies - shifts1 - fsr * setup.calibration.orders
        d2 = np.array([frequencies[1] - frequencies[0] - (frequencies[-1] - frequencies[-2]), frequencies[2] - frequencies[0] - (frequencies[-1] - frequencies[-3])])
        d3 = np.array([shifts1[1] - frequencies[1] + frequencies[0], shifts1[2] - frequencies[2] + frequencies[0], shifts1[2] - frequencies[5] + frequencies[3], shifts1[1] - frequencies[5] + frequencies[4]])
        return np.sum(d1 ** 2) + np.sum(d2 ** 2) + np.sum(d3 ** 2)
    opt_result = fmin(error, vipa_start, args=(peaks, setup.calibration.shifts))
    return (opt_result[0], opt_result[1], opt_result[2], opt_result[3])

def VIPA(x, vipa_params):
    """
    Returns the absolute frequency in Hz of a given point
    on the spectrum in pixels.
    Subtract the absolute laser frequency from this value
    to get the relative shift.

    Parameters
    ----------
    x:   [pix]   The point on the spectrum
    vipa_params: Fit parameters of the VIPA fit

    Returns
    -------
    [Hz]   The frequency of the given point on the spectrum
    """
    return 1.0 / (vipa_params[0] + vipa_params[1] * x + vipa_params[2] * x ** 2)

def arc_coordinates(points,shape=(400,400),width=2,num_points=500):
    """Circle fit, original 64-edge boundary approximation, angle-uniform strip.

    GEOS/Shapely's default circle buffer uses 16 edges per quadrant. Explicit
    segment/rectangle intersections preserve this discretization locally.
    Coordinates are row,column; shape edges lie at 0 and 400 as upstream.
    """
    center,radius=fit_circle([tuple(p) for p in points]);center=np.asarray(center)
    angles=-np.arange(65)*2*np.pi/64
    polygon=center+radius*np.stack([np.cos(angles),np.sin(angles)],axis=-1)
    intersections=[]
    for a,b in zip(polygon[:-1],polygon[1:]):
        for axis in [0,1]:
            other=1-axis
            for edge in [0,shape[axis]]:
                if b[axis]!=a[axis]:
                    t=(edge-a[axis])/(b[axis]-a[axis])
                    if 0<=t<=1:
                        p=a+t*(b-a)
                        if 0<=p[other]<=shape[other]:intersections.append(p)
    intersections.sort(key=lambda p:p[0])
    phis=[np.arctan2(*(p-center)[::-1])%(2*np.pi) for p in intersections]
    phi=np.linspace(phis[0],phis[-1],num_points)
    unit=np.stack([np.cos(phi),np.sin(phi)],axis=-1)
    # Same arithmetic order as source: center + radius*unit, then radial offset.
    arc=np.empty((num_points,2*width+1,2))
    for i,e in enumerate(unit):arc[i]=center+radius*e+e*np.arange(-width,width+1)[:,None]
    return arc,np.r_[center,radius]

def spectral_model(x,parameters):
    """One Lorentzian: center, FWHM, intensity, additive background."""
    return lorentz(x,*parameters[:3])+parameters[3]

def vipa_objective(params,peaks,instrument=None):
    instrument=setup() if instrument is None else instrument
    f=VIPA(peaks,params)-instrument.f0;s=instrument.calibration.shifts
    d1=f-s-params[3]*instrument.calibration.orders
    d2=np.array([(f[1]-f[0])-(f[-1]-f[-2]),(f[2]-f[0])-(f[-1]-f[-3])])
    d3=np.array([s[1]-f[1]+f[0],s[2]-f[2]+f[0],s[2]-f[5]+f[3],s[1]-f[5]+f[4]])
    return np.sum(d1**2)+np.sum(d2**2)+np.sum(d3**2)
