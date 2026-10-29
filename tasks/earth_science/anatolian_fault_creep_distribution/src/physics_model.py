"""Closed elastic surface Green functions, locally executable equations.

Extracted from CSI commit074ca028697b488fdb036e67d799787b374ddbb9,
triangularDisp.py displacement/RotateXyVec/adv (GPL-3.0); author Meade2007
angular-dislocation construction. No CSI imports or domain-library calls.
Coordinates km, depth positive down; returned displacement is east,north,up
per slip unit. The selected acquisition is surface-only, never interior.
"""
import copy
import numpy as np

POISSON_RATIO = 0.25
SHEAR_MODULUS_PA = 30e9
BASIN_CENTER_KM = (-30.343562481406668, 8.946455159060937)
BASIN_DEPTH_M = 3000.0
BASIN_RADIUS_M = 6000.0


def mogi_deflation_basis(xy_km):
    """ENU mm per formal kPa deflation (empirical Mogi-shaped nuisance).

    This fixed shape uses radius6km/depth3km and violates the small-cavity
    approximation; its coefficient must not be interpreted as geological
    pressure truth. Units close explicitly: m/Pa *1000mm/m *1000Pa/kPa.
    The original tutorial -1e6 scaling has exactly this numerical meaning.
    """
    delta = (np.asarray(xy_km, dtype=float)-np.array(BASIN_CENTER_KM))*1000.
    r = np.sqrt(BASIN_DEPTH_M**2 + np.sum(delta**2, axis=1))
    # Cartesian expression equals source polar calculation to roundoff.
    v = np.column_stack((delta, np.full(len(delta), BASIN_DEPTH_M)))
    return -1e6*BASIN_RADIUS_M**3*(1.-POISSON_RATIO)*v/(SHEAR_MODULUS_PA*r[:, None]**3)


def build_green_matrix(data):
    """Return3554x1305 matrix:1299slips, basin kPa/year, five mm/year datums.

    Slip order:1292NAF triangles,5deep rectangles,east,west. Rectangles use
    the exact two-triangle identity checked against original Okada formulas.
    Positive slip is the tutorial right-lateral convention, hence -unit ss.
    """
    xy = np.concatenate((data['sar_xy'], data['gnss_xy']))
    x, y = xy.T
    zero = np.zeros(len(xy))
    n_sar = len(data['sar_xy'])

    def project(enu):
        return np.concatenate((np.sum(enu[:n_sar]*data['sar_los'], axis=1),
                               enu[n_sar:, :2].T.ravel()))

    matrix = np.zeros((n_sar+78, 1305), dtype=float)
    for j, triangle in enumerate(data['fault_triangles']):
        enu = np.column_stack(displacement(x, y, zero, list(triangle), 1., 0., 0.))
        matrix[:, j] = -project(enu)
    rectangles = np.concatenate((data['deep_rectangles'], data['remote_rectangles']))
    for j, rectangle in enumerate(rectangles):
        enu = sum(np.column_stack(displacement(x, y, zero, list(t), 1., 0., 0.))
                  for t in [rectangle[[0,1,2]], rectangle[[0,2,3]]])
        matrix[:, 1292+j] = -project(enu)
    matrix[:, 1299] = project(mogi_deflation_basis(xy))
    for track in range(3):
        matrix[:n_sar, 1300+track] = data['sar_track_index'] == track
    matrix[n_sar:n_sar+39, 1303] = 1.
    matrix[n_sar+39:, 1304] = 1.
    return matrix


def build_data_covariance(data, diagonal=False):
    """Block noise covariance, squared mm/year; exact known source noise model."""
    from .preprocessing import SAR_COUNTS, NOISE_SIGMA, NOISE_LENGTH_KM, NOISE_STD
    covariance = np.zeros((3554, 3554))
    start = 0
    for count, sigma, length, std in zip(SAR_COUNTS, NOISE_SIGMA, NOISE_LENGTH_KM, NOISE_STD):
        xy = data['sar_xy'][start:start+count]
        distance = np.sqrt((xy[:,None,0]-xy[None,:,0])**2+(xy[:,None,1]-xy[None,:,1])**2)
        block = sigma*sigma*np.exp(-distance/length)
        np.fill_diagonal(block, std**2)
        if diagonal:
            block = np.diag(np.diag(block))
        covariance[start:start+count,start:start+count] = block
        start += count
    covariance[start:,start:] = np.diag(data['gnss_std'].T.ravel()**2)
    return covariance


def build_model_covariance(triangles, sigma=20., lengths_km=(10., 1.)):
    """Source conditional covariance: actual0.5decay and unsquared diagonal inputs.

    The source buildCmXY docstring disagrees with its executable formula.
    This explicit function preserves executable author tutorial, not that text.
    """
    centers = (triangles[:,0]+triangles[:,1]+triangles[:,2])/3.
    spacing = np.array([np.ptp(np.unique(centers[:,j]))/len(np.unique(centers[:,j])) for j in range(3)])
    lam0 = np.sqrt(np.sum(spacing**2))
    lx, lz = lengths_km
    h = np.sqrt((centers[:,None,0]-centers[None,:,0])**2+(centers[:,None,1]-centers[None,:,1])**2)
    v = abs(centers[:,None,2]-centers[None,:,2])
    covariance = np.zeros((1305,1305))
    covariance[:1292,:1292] = (sigma*lam0/lx)*(sigma*lam0/lz)*np.exp(-.5*h/lx)*np.exp(-.5*v/lz)
    covariance[1292:,1292:] = np.diag([100.]*5+[.01,.01,1.]+[1.]*5)
    return covariance

def displacement(sx, sy, sz, vertices, ss, ds, ts, nu=0.25):
    """
    Computes the displacement vector at an observation point due to slip on one
    triangular patch at depth.

    Args:
        * sx        : x coordinates of ground points
        * sy        : y coordinates of ground points
        * sz        : z coordinates of ground points
        * vertices  : list of 3-component vertices of the triangle
        * ss        : amount of strike-slip
        * ds        : amount of dip-slip
        * ts        : amount of tensile/opening-slip
        * nu        : Poisson's ratio
    
    Returns:
        * ux        : x component of displacement
        * uy        : y component of displacement
        * uz        : z component of displacement
    """

    # Modify input vertices by appending first point to end
    verts = copy.deepcopy(vertices)
    verts.append(verts[0])
    verts = [np.array(p) for p in verts]
    p1, p2, p3 = verts[:3]

    # Compute normal vector
    dx1 = p2 - p1
    dx2 = p3 - p1 
    normVec = np.cross(dx1, dx2)
    normVec /= np.linalg.norm(normVec)
    # Enforce clockwise circulation
    if normVec[2] < 0: 
        normVec *= -1.0
        # Update vertices by swapping 2nd and 3rd vertices
        verts = [p1, p3, p2, p1]
    
    # Compute slip vector in global coordinates
    angle = np.arctan2(normVec[1], normVec[0])
    strikeVec = np.array([-np.sin(angle), np.cos(angle), 0.0])
    dipVec = np.cross(normVec, strikeVec)
    # Change sign of slip components to be consistent with Okada (BVR)
    slipComp = np.array([-ss, -ds, -ts])
    slipVec = np.dot(np.column_stack((strikeVec, dipVec, normVec)), slipComp)

    # Allocate solution vectors
    ux = np.zeros_like(sx)
    uy = np.zeros_like(sx)
    uz = np.zeros_like(sx)
    
    # Loop over legs of triangle
    halfPi = 0.5 * np.pi
    for iTri in range(3):
    
        # Get the vertices of the leg
        p1, p2 = verts[iTri], verts[iTri+1]
        x1, y1, z1 = p1
        x2, y2, z2 = p2
    
        # Calculate strike and dip of current leg
        dx, dy = x2 - x1, y2 - y1
        strike = np.arctan2(dy, dx)
        segMapLength = np.sqrt(dx**2 + dy**2)
        rx, ry = RotateXyVec(dx, dy, -strike)
        dip = np.arctan2(z2 - z1, rx)
    
        if dip >= 0.0:
            beta = halfPi - dip
            if beta > halfPi:
                beta = halfPi - beta
        else:
            beta = -1.0 * (halfPi + dip)
            if beta < -halfPi:
                beta = halfPi - abs(beta)

        ssVec = np.array([np.cos(strike), np.sin(strike), 0.0])
        tsVec = np.array([-np.sin(strike), np.cos(strike), 0.0])
        dsVec = np.cross(ssVec, tsVec)
        lss = np.dot(slipVec, ssVec)
        lts = np.dot(slipVec, tsVec)
        lds = np.dot(slipVec, dsVec)
        
        if (abs(beta) > 1.0e-6) and (abs(beta - np.pi) > 1.0e-6):
            # First angular dislocation
            sx1, sy1 = RotateXyVec(sx - x1, sy - y1, -strike)
            ux1, uy1, uz1 = adv(sx1, sy1, sz - z1, z1, beta, nu, lss, lts, lds)

            # Second angular dislocation
            sx2, sy2 = RotateXyVec(sx - x2, sy - y2, -strike)
            ux2, uy2, uz2 = adv(sx2, sy2, sz - z2, z2, beta, nu, lss, lts, lds)
    
            # Rotate vectors to correct for strike
            uxn, uyn = RotateXyVec(ux1 - ux2, uy1 - uy2, strike)
            uzn = uz1 - uz2
         
            # Add the displacements from current leg
            ux += uxn
            uy += uyn
            uz += uzn

    # This task observes the free surface only (sz=0), with all vertices at
    # nonnegative depth. There are no observation points below a buried patch,
    # so the source internal-prism solid translation is identically inactive.
    # Make sure to change sign on the vertical displacements!
    return ux, uy, -uz

def RotateXyVec(x, y, alpha):
    """
    Rotate components by an angle alpha.
    """
    sina, cosa = np.sin(alpha), np.cos(alpha)
    xp = x * cosa - y * sina + np.spacing(1)
    yp = x * sina + y * cosa + np.spacing(1)
    return xp, yp

def adv(y1, y2, y3, a, beta, nu, B1, B2, B3):
    """    
    These are the displacements in a uniform elastic half space due to slip
    on an angular dislocation (Comninou and Dunders, 1975).  Some of the
    equations for the B2 and B3 cases have been corrected following Thomas
    1993.  The equations are coded in way such that they roughly correspond
    to each line in original text.  Exceptions have been made where it made 
    more sense because of grouping symbols.
    """

    sinbeta = np.sin(beta)
    cosbeta = np.cos(beta)
    cotbeta = 1.0 / np.tan(beta)
    z1 = y1 * cosbeta - y3 * sinbeta
    z3 = y1 * sinbeta + y3 * cosbeta
    R2 = y1 * y1 + y2 * y2 + y3 * y3
    R = np.sqrt(R2)
    y3bar = y3 + 2.0 * a
    z1bar =  y1 * cosbeta + y3bar * sinbeta
    z3bar = -y1 * sinbeta + y3bar * cosbeta
    R2bar = y1 * y1 + y2 * y2 + y3bar * y3bar
    Rbar = np.sqrt(R2bar)
    F = (-np.arctan2(y2, y1) + np.arctan2(y2, z1) 
        + np.arctan2(y2*R*sinbeta, y1*z1 + y2*y2*cosbeta))
    Fbar = (-np.arctan2(y2, y1) + np.arctan2(y2, z1bar) 
           + np.arctan2(y2*Rbar*sinbeta, y1*z1bar + y2*y2*cosbeta))
    Rbar3 = Rbar**3
    cotbeta2 = cotbeta**2
    Rby3bar = Rbar + y3bar
    Rbz3bar = Rbar + z3bar

    # Cache Poisson's ratio terms
    nu1 = 1.0 - nu
    nu2 = 1.0 - 2.0 * nu
    
    # Case I: Burgers vector (B1,0,0)
    v1InfB1 = (2.0 * nu1 * (F + Fbar) - y1 * y2 * (1.0 / (R * (R - y3))
             + 1.0 / (Rbar * (Rby3bar))) - y2 * cosbeta * ((R * sinbeta - y1) 
             / (R * (R-z3)) + (Rbar * sinbeta - y1) / (Rbar * (Rbz3bar))))
    v2InfB1 = (nu2 * (np.log(R - y3) + np.log(Rby3bar) 
              - cosbeta * (np.log(R - z3) + np.log(Rbz3bar))) 
              - y2 * y2 * (1.0 / (R * (R - y3)) + 1.0 / (Rbar * (Rby3bar)) 
              - cosbeta * (1.0 / (R * (R - z3)) + 1.0 / (Rbar * (Rbz3bar)))))
    v3InfB1 = (y2 * (1.0 / R - 1.0 / Rbar - cosbeta * ((R * cosbeta - y3) / (R * (R - z3)) 
            - (Rbar * cosbeta + y3bar) / (Rbar * (Rbz3bar)))))
    denom = 1.0 / (8.0 * np.pi * nu1)
    v1InfB1 *= denom
    v2InfB1 *= denom
    v3InfB1 *= denom
    
    v1CB1 = (-2.0 * nu1 * nu2 * Fbar * (cotbeta2) 
           + nu2 * y2 / (Rby3bar) * ((1.0 - 2.0 * nu - a / Rbar) * cotbeta 
           - y1 / (Rby3bar) * (nu + a / Rbar)) + nu2 * y2 * cosbeta * cotbeta 
           / (Rbz3bar) * (cosbeta + a / Rbar) + a * y2 * (y3bar - a) * cotbeta 
           / Rbar3 + y2 * (y3bar - a) / (Rbar * (Rby3bar)) * (-(1.0 - 2.0*nu)
           * cotbeta + y1 / (Rby3bar) * (2.0 * nu + a / Rbar) + a * y1 / (Rbar * Rbar)) 
           + y2 * (y3bar - a) / (Rbar * (Rbz3bar)) * (cosbeta / (Rbz3bar) 
           * ((Rbar * cosbeta + y3bar) * (nu2 * cosbeta - a / Rbar) * cotbeta 
           + 2.0 * nu1 * (Rbar * sinbeta - y1) * cosbeta) - a * y3bar *cosbeta *cotbeta 
           / (Rbar * Rbar)))

    v2CB1 = (nu2 * ((2.0 * nu1 * (cotbeta2) - nu) 
           * np.log(Rby3bar) - (2.0 * nu1 * (cotbeta2) + 1.0 - 2.0 * nu) 
           * cosbeta * np.log(Rbz3bar)) - nu2 / (Rby3bar) * (y1 * cotbeta
           * (1.0 - 2.0 * nu - a / Rbar) + nu * y3bar - a + (y2 * y2) / (Rby3bar) 
           * (nu + a / Rbar)) - nu2 * z1bar * cotbeta / (Rbz3bar) * (cosbeta + a / Rbar) 
           - a * y1 * (y3bar - a) * cotbeta / Rbar3 + (y3bar - a) / (Rby3bar)
           * (-2.0 * nu + 1.0 / Rbar * (nu2 * y1 * cotbeta - a) + (y2 * y2) 
           / (Rbar * (Rby3bar)) * (2.0 * nu + a / Rbar) + a * (y2 * y2) 
           / Rbar3) + (y3bar - a) / (Rbz3bar) * ((cosbeta * cosbeta) - 1.0 
           / Rbar * (nu2 * z1bar * cotbeta + a * cosbeta) + a * y3bar * z1bar 
           * cotbeta / Rbar3 - 1.0 / (Rbar * (Rbz3bar)) * ((y2 * y2) 
           * (cosbeta * cosbeta) - a * z1bar * cotbeta / Rbar * (Rbar * cosbeta + y3bar))))
    
    v3CB1 = (2.0 * nu1 * ((nu2 * Fbar * cotbeta) + (y2 / (Rby3bar) 
          * (2.0 * nu + a / Rbar)) - (y2 * cosbeta / (Rbz3bar) * (cosbeta + a / Rbar))) 
          + y2 * (y3bar - a) / Rbar * (2 * nu / (Rby3bar) + a / (Rbar * Rbar)) + y2 
          * (y3bar - a) * cosbeta / (Rbar * (Rbz3bar)) * (1.0 -2.0 * nu - (Rbar * cosbeta
          + y3bar) / (Rbz3bar) * (cosbeta + a / Rbar) - a * y3bar / (Rbar * Rbar)))
   
    denom = 1.0 / (4.0 * np.pi * nu1) 
    v1CB1 *= denom 
    v2CB1 *= denom 
    v3CB1 *= denom 
    
    v1B1 = v1InfB1 + v1CB1
    v2B1 = v2InfB1 + v2CB1
    v3B1 = v3InfB1 + v3CB1

    # Case II: Burgers vector (0,B2,0)
    v1InfB2 = (-nu2 * (np.log(R - y3) + np.log(Rby3bar) - cosbeta 
            * (np.log(R - z3) + np.log(Rbz3bar))) + y1 * y1 * (1.0 / (R * (R - y3)) + 1.0 
            / (Rbar * (Rby3bar))) + z1 * (R * sinbeta - y1) / (R * (R - z3)) + z1bar 
            * (Rbar * sinbeta - y1) / (Rbar * (Rbz3bar)))
    v2InfB2 = (2.0 * nu1 * (F + Fbar) + y1 * y2 * (1.0 / (R * (R - y3)) + 1.0 / (Rbar 
            * (Rby3bar))) - y2 * (z1 / (R * (R - z3)) + z1bar / (Rbar * (Rbz3bar))))
    v3InfB2 = (-nu2 * sinbeta * (np.log(R - z3) - np.log(Rbz3bar)) - y1 
            * (1.0 / R - 1.0 / Rbar) + z1 * (R * cosbeta - y3) / (R * (R - z3)) - z1bar 
            * (Rbar * cosbeta + y3bar) / (Rbar * (Rbz3bar)))
    denom = 1.0 / (8.0 * np.pi * nu1)
    v1InfB2 *= denom
    v2InfB2 *= denom
    v3InfB2 *= denom
    
    v1CB2 = (nu2 * ((2.0 * nu1 * (cotbeta2) + nu) 
          * np.log(Rby3bar) - (2.0 * nu1 * (cotbeta2) + 1.0) * cosbeta
          * np.log(Rbz3bar)) + nu2 / (Rby3bar) * (-nu2 
          * y1 * cotbeta + nu * y3bar - a + a * y1 * cotbeta / Rbar + (y1 * y1) / (Rbar 
          + y3bar) * (nu + a / Rbar)) - nu2 * cotbeta / (Rbz3bar) * (z1bar * cosbeta 
          - a * (Rbar * sinbeta - y1) / (Rbar * cosbeta)) - a * y1 * (y3bar - a) * cotbeta 
          / Rbar3 + (y3bar - a) / (Rby3bar) * (2.0 * nu + 1.0 / Rbar 
          * (nu2 * y1 * cotbeta + a) - (y1 * y1) / (Rbar * (Rby3bar)) 
          * (2.0 * nu + a / Rbar) - a * (y1 * y1) / Rbar3) + (y3bar - a) * cotbeta
          / (Rbz3bar) * (-cosbeta * sinbeta + a * y1 * y3bar / (Rbar3 * cosbeta) 
          + (Rbar * sinbeta - y1) / Rbar * (2.0 * nu1 * cosbeta - (Rbar * cosbeta + y3bar) 
          / (Rbar+z3bar) * (1.0 + a / (Rbar * cosbeta)))))

    v2CB2 = (2.0 * nu1 * nu2 * Fbar * cotbeta2 + (1.0 - 2 * nu) 
          * y2 / (Rby3bar) * (-(1.0 - 2.0 * nu - a / Rbar) * cotbeta + y1 / (Rby3bar)
          * (nu + a / Rbar)) - nu2 * y2 * cotbeta / (Rbz3bar) * (1.0 + a 
          / (Rbar * cosbeta)) - a * y2 * (y3bar - a) * cotbeta / Rbar3 + y2 
          * (y3bar - a) / (Rbar * (Rby3bar)) * (nu2 * cotbeta - 2.0 * nu * y1 
          / (Rby3bar) - a * y1 / Rbar * (1.0 / Rbar + 1.0 / (Rby3bar))) + y2 
          * (y3bar - a) * cotbeta / (Rbar * (Rbz3bar)) * (-2.0 * nu1 * cosbeta 
          + (Rbar * cosbeta + y3bar) / (Rbz3bar) * (1.0 + a / (Rbar * cosbeta)) + a 
          * y3bar / ((Rbar * Rbar) * cosbeta)))

    v3CB2 = (-2.0 * nu1 * nu2 * cotbeta * (np.log(Rby3bar) - cosbeta
          * np.log(Rbz3bar)) - 2.0 * nu1 * y1 / (Rby3bar) * (2.0 * nu + a / Rbar) 
          + 2.0 * nu1 * z1bar / (Rbz3bar) * (cosbeta + a / Rbar) + (y3bar - a) / Rbar
          * (nu2 * cotbeta - 2.0 * nu * y1 / (Rby3bar) - a * y1 / (Rbar * Rbar)) 
          - (y3bar - a) / (Rbz3bar) * (cosbeta * sinbeta + (Rbar * cosbeta + y3bar) 
          * cotbeta / Rbar * (2.0 * nu1 * cosbeta - (Rbar * cosbeta + y3bar) / (Rbz3bar)) 
          + a / Rbar * (sinbeta - y3bar * z1bar / (Rbar * Rbar) - z1bar * (Rbar * cosbeta 
          + y3bar) / (Rbar * (Rbz3bar)))))

    denom = 1.0 / (4.0 * np.pi * nu1)
    v1CB2 *= denom
    v2CB2 *= denom
    v3CB2 *= denom
    v1B2 = v1InfB2 + v1CB2
    v2B2 = v2InfB2 + v2CB2
    v3B2 = v3InfB2 + v3CB2

    # Case III: Burgers vector (0,0,B3)
    v1InfB3 = (y2 * sinbeta * ((R * sinbeta - y1) / (R * (R - z3)) + (Rbar * sinbeta - y1) 
            / (Rbar * (Rbz3bar))))
    v2InfB3 = (nu2 * sinbeta * (np.log(R - z3) + np.log(Rbz3bar)) - (y2 * y2) 
            * sinbeta * (1.0 / (R * (R - z3)) + 1.0 / (Rbar * (Rbz3bar))))
    v3InfB3 = (2.0 * nu1 * (F - Fbar) + y2 * sinbeta * ((R * cosbeta - y3) / (R 
            * (R - z3)) - (Rbar * cosbeta + y3bar) / (Rbar * (Rbz3bar))))
    denom = 1.0 / (8.0 * np.pi * nu1)
    v1InfB3 *= denom
    v2InfB3 *= denom
    v3InfB3 *= denom
    
    v1CB3 = (nu2 * (y2 / (Rby3bar) * (1.0 + a / Rbar) - y2 * cosbeta / (Rbar
          + z3bar) * (cosbeta + a / Rbar)) - y2 * (y3bar - a) / Rbar * (a / (Rbar * Rbar) 
          + 1.0 / (Rby3bar)) + y2 * (y3bar - a) * cosbeta / (Rbar * (Rbz3bar)) 
          * ((Rbar * cosbeta + y3bar) / (Rbz3bar) * (cosbeta + a / Rbar) + a * y3bar 
          / (Rbar * Rbar)))

    v2CB3 = (nu2 * (-sinbeta * np.log(Rbz3bar) - y1 / (Rby3bar) * (1.0 
          + a / Rbar) + z1bar / (Rbz3bar) * (cosbeta + a / Rbar)) + y1 * (y3bar - a) 
          / Rbar * (a / (Rbar * Rbar) + 1.0 / (Rby3bar)) - (y3bar - a) / (Rbz3bar) 
          * (sinbeta * (cosbeta - a / Rbar) + z1bar / Rbar * (1 + a * y3bar / (Rbar * Rbar)) 
          - 1.0 / (Rbar * (Rbz3bar)) * ((y2 * y2) * cosbeta * sinbeta - a * z1bar / Rbar 
          * (Rbar * cosbeta + y3bar))))

    v3CB3 = (2.0 * nu1 * Fbar + 2.0 * nu1 * (y2 * sinbeta / (Rbz3bar) 
          * (cosbeta + a / Rbar)) + y2 * (y3bar - a) * sinbeta / (Rbar * (Rbz3bar)) 
          * (1.0 + (Rbar * cosbeta + y3bar) / (Rbz3bar) * (cosbeta + a / Rbar) + a 
          * y3bar / (Rbar * Rbar)))

    denom = 1.0 / (4.0 * np.pi * nu1)
    v1CB3 *= denom
    v2CB3 *= denom
    v3CB3 *= denom
    
    v1B3 = v1InfB3 + v1CB3
    v2B3 = v2InfB3 + v2CB3
    v3B3 = v3InfB3 + v3CB3

    # Sum the for each slip component
    v1 = B1 * v1B1 + B2 * v1B2 + B3 * v1B3
    v2 = B1 * v2B1 + B2 * v2B2 + B3 * v2B3
    v3 = B1 * v3B1 + B2 * v3B2 + B3 * v3B3

    return v1, v2, v3
