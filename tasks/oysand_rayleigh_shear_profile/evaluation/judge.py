"""Explicit Dunkin compound matrices and bracketed phase-velocity root search.

Extracted from disba0.7.0 surf96, BSD3-Clause, copyright2020Keurfon Luu.
CPS ancestry copyright1986,1991D.R.Russell,R.B.Herrmann, Saint Louis University.
Only the selected dry-solid fundamental Rayleigh phase path is shipped.
No compiled/domain solver, JIT decorator or library reconstruction is imported.
"""
import numpy as np

twopi=2*np.pi
class DispersionError(ValueError):pass


def normc(ee, nmat):
    """Normalize Haskell or Dunkin vectors."""
    t1 = 0.0
    for i in range(nmat):
        t1 = max(t1, np.abs(ee[i]))

    if t1 < 1.0e-40:
        t1 = 1.0

    for i in range(nmat):
        ee[i] /= t1

    ex = np.log(t1)
    return ee, ex

def dnka(wvno2, gam, gammk, rho, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz, ca):
    """Dunkin's matrix."""
    gamm1 = gam - 1.0
    twgm1 = gam + gamm1
    gmgmk = gam * gammk
    gmgm1 = gam * gamm1
    gm1sq = gamm1 * gamm1

    rho2 = rho * rho
    a0pq = a0 - cpcq
    t = -2.0 * wvno2

    ca[0, 0] = cpcq - 2.0 * gmgm1 * a0pq - gmgmk * xz - wvno2 * gm1sq * wy
    ca[0, 1] = (wvno2 * cpy - cqx) / rho
    ca[0, 2] = -(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho
    ca[0, 3] = (cpz - wvno2 * cqw) / rho
    ca[0, 4] = -(2.0 * wvno2 * a0pq + xz + wvno2 * wvno2 * wy) / rho2

    ca[1, 0] = (gmgmk * cpz - gm1sq * cqw) * rho
    ca[1, 1] = cpcq
    ca[1, 2] = gammk * cpz - gamm1 * cqw
    ca[1, 3] = -wz
    ca[1, 4] = ca[0, 3]

    ca[3, 0] = (gm1sq * cpy - gmgmk * cqx) * rho
    ca[3, 1] = -xy
    ca[3, 2] = gamm1 * cpy - gammk * cqx
    ca[3, 3] = ca[1, 1]
    ca[3, 4] = ca[0, 1]

    ca[4, 0] = (
        -(2.0 * gmgmk * gm1sq * a0pq + gmgmk * gmgmk * xz + gm1sq * gm1sq * wy) * rho2
    )
    ca[4, 1] = ca[3, 0]
    ca[4, 2] = (
        -(gammk * gamm1 * twgm1 * a0pq + gam * gammk * gammk * xz + gamm1 * gm1sq * wy)
        * rho
    )
    ca[4, 3] = ca[1, 0]
    ca[4, 4] = ca[0, 0]

    ca[2, 0] = t * ca[4, 2]
    ca[2, 1] = t * ca[3, 2]
    ca[2, 2] = a0 + 2.0 * (cpcq - ca[0, 0])
    ca[2, 3] = t * ca[1, 2]
    ca[2, 4] = t * ca[0, 2]

    return ca

def var(p, q, ra, rb, wvno, xka, xkb, dpth):
    """Find variables cosP, cosQ, sinP, sinQ..."""
    # Examine P-wave eigenfunctions
    # Checking whether c > vp, c = vp or c < vp
    pex = 0.0
    if wvno < xka:
        sinp = np.sin(p)
        w = sinp / ra
        x = -ra * sinp
        cosp = np.cos(p)
    elif wvno == xka:
        cosp = 1.0
        w = dpth
        x = 0.0
    elif wvno > xka:
        pex = p
        fac = np.exp(-2.0 * p) if p < 16.0 else 0.0
        cosp = (1.0 + fac) * 0.5
        sinp = (1.0 - fac) * 0.5
        w = sinp / ra
        x = ra * sinp

    # Examine S-wave eigenfunctions
    # Checking whether c > vs, c = vs or c < vs
    sex = 0.0
    if wvno < xkb:
        sinq = np.sin(q)
        y = sinq / rb
        z = -rb * sinq
        cosq = np.cos(q)
    elif wvno == xkb:
        cosq = 1.0
        y = dpth
        z = 0.0
    elif wvno > xkb:
        sex = q
        fac = np.exp(-2.0 * q) if q < 16.0 else 0.0
        cosq = (1.0 + fac) * 0.5
        sinq = (1.0 - fac) * 0.5
        y = sinq / rb
        z = rb * sinq

    # Form eigenfunction products for use with compound matrices
    exa = pex + sex
    a0 = np.exp(-exa) if exa < 60.0 else 0.0
    cpcq = cosp * cosq
    cpy = cosp * y
    cpz = cosp * z
    cqw = cosq * w
    cqx = cosq * x
    xy = x * y
    xz = x * z
    wy = w * y
    wz = w * z

    qmp = sex - pex
    fac = np.exp(qmp) if qmp > -40.0 else 0.0
    cosq *= fac
    y *= fac
    z *= fac

    return w, cosp, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz

def dltar4(wvno, omega, d, a, b, rho, llw, ca):
    """Rayleigh-wave period equation."""
    e = np.zeros(5, dtype=np.float64)
    ee = np.zeros(5, dtype=np.float64)

    omega = max(omega, 1.0e-4)
    wvno2 = wvno * wvno
    xka = omega / a[-1]
    xkb = omega / b[-1]
    wvnop = wvno + xka
    wvnom = np.abs(wvno - xka)
    ra = np.sqrt(wvnop * wvnom)
    wvnop = wvno + xkb
    wvnom = np.abs(wvno - xkb)
    rb = np.sqrt(wvnop * wvnom)
    t = b[-1] / omega

    # E matrix for the bottom half-space
    gammk = 2.0 * t * t
    gam = gammk * wvno2
    gamm1 = gam - 1.0
    rho1 = rho[-1]
    e[0] = rho1 * rho1 * (gamm1 * gamm1 - gam * gammk * ra * rb)
    e[1] = -rho1 * ra
    e[2] = rho1 * (gamm1 - gammk * ra * rb)
    e[3] = rho1 * rb
    e[4] = wvno2 - ra * rb

    # Matrix multiplication from bottom layer upward
    for m in range(len(d) - 2, llw, -1):
        xka = omega / a[m]
        xkb = omega / b[m]
        t = b[m] / omega
        gammk = 2.0 * t * t
        gam = gammk * wvno2
        wvnop = wvno + xka
        wvnom = np.abs(wvno - xka)
        ra = np.sqrt(wvnop * wvnom)
        wvnop = wvno + xkb
        wvnom = np.abs(wvno - xkb)
        rb = np.sqrt(wvnop * wvnom)

        dpth = d[m]
        rho1 = rho[m]
        p = ra * dpth
        q = rb * dpth

        # Evaluate cosP, cosQ...
        _, _, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz = var(
            p, q, ra, rb, wvno, xka, xkb, dpth
        )

        # Evaluate Dunkin's matrix
        ca = dnka(
            wvno2, gam, gammk, rho1, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz, ca
        )

        for i in range(5):
            ee[i] = 0.0
            for j in range(5):
                ee[i] += e[j] * ca[j, i]

        ee, _ = normc(ee, 5)
        for i in range(5):
            e[i] = ee[i]

    if llw == 0:
        xka = omega / a[0]
        wvnop = wvno + xka
        wvnom = np.abs(wvno - xka)
        ra = np.sqrt(wvnop * wvnom)
        dpth = d[0]
        rho1 = rho[0]
        p = ra * dpth
        w, cosp, _, _, _, _, _, _, _, _, _, _ = var(
            p, q, ra, 1.0e-5, wvno, xka, xkb, dpth
        )
        dlt = cosp * e[0] - rho1 * w * e[1]
    else:
        dlt = e[0]

    return dlt

def nevill(t, c1, c2, del1, del2, d, a, b, rho, ifunc, llw, ca):
    """Hybrid method for refining root once it has been bracketted."""
    x = np.zeros(20, dtype=np.float64)
    y = np.zeros(20, dtype=np.float64)

    # Initial guess
    omega = twopi / t
    c3 = 0.5 * (c1 + c2)
    del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw, ca)
    nev = 1
    nctrl = 1

    while True:
        nctrl += 1
        if nctrl >= 100:
            break

        # Make sure new estimate is inside the previous values
        # If not, perform interval halving
        if c3 < min(c1, c2) or c3 > max(c1, c2):
            nev = 0
            c3 = 0.5 * (c1 + c2)
            del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw, ca)

        s13 = del1 - del3
        s32 = del3 - del2

        # Define new bounds according to the sign of the period equation
        if np.sign(del3) * np.sign(del1) < 0.0:
            c2 = c3
            del2 = del3
        else:
            c1 = c3
            del1 = del3

        # Check for convergence
        if np.abs(c1 - c2) <= 1.0e-6 * c1:
            break

        # If the slopes are not the same between c1, c2 and c3
        # Do not use Neville iteration
        if np.sign(s13) != np.sign(s32):
            nev = 0

        # If the period equation differs by more than a factor of 10
        # Use interval halving to avoid poor behavior of polynomial fit
        ss1 = np.abs(del1)
        s1 = 0.01 * ss1
        ss2 = np.abs(del2)
        s2 = 0.01 * ss2
        if s1 > ss2 or s2 > ss1 or nev == 0:
            c3 = 0.5 * (c1 + c2)
            del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw, ca)
            nev = 1
            m = 1
        else:
            if nev == 2:
                x[m - 1] = c3
                y[m - 1] = del3
            else:
                x[0] = c1
                y[0] = del1
                x[1] = c2
                y[1] = del2
                m = 1

            # Perform Neville iteration
            flag = 1
            for kk in range(m):
                j = m - kk
                denom = y[m] - y[j]
                if np.abs(denom) < 1.0e-10 * np.abs(y[m]):
                    flag = 0
                    break
                else:
                    x[j - 1] = (-y[j - 1] * x[j] + y[m] * x[j - 1]) / denom

            if flag:
                c3 = x[0]
                del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw, ca)
                nev = 2
                m += 1
                m = min(m, 10)
            else:
                c3 = 0.5 * (c1 + c2)
                del3 = dltar(omega / c3, omega, d, a, b, rho, ifunc, llw, ca)
                nev = 1
                m = 1

    return c3

def getsol(t1, c1, clow, dc, cm, betmx, ifirst, del1st, d, a, b, rho, ifunc, llw, ca):
    """Bracket dispersion curve and then refine it."""
    # Bracket solution
    omega = twopi / t1
    del1 = dltar(omega / c1, omega, d, a, b, rho, ifunc, llw, ca)
    del1st = del1 if ifirst else del1st
    idir = -1.0 if not ifirst and np.sign(del1st) * np.sign(del1) < 0.0 else 1.0

    # idir indicates the direction of the search for the true phase velocity from the initial estimate
    while True:
        c2 = c1 + idir * dc

        if c2 <= clow:
            idir = 1.0
            c1 = clow
        else:
            omega = twopi / t1
            del2 = dltar(omega / c2, omega, d, a, b, rho, ifunc, llw, ca)

            if np.sign(del1) != np.sign(del2):
                c1 = nevill(t1, c1, c2, del1, del2, d, a, b, rho, ifunc, llw, ca)
                iret = c1 > betmx
                break

            c1 = c2
            del1 = del2

            iret = c1 < cm or c1 >= betmx + dc
            if iret:
                break

    return c1, del1st, iret

def gtsolh(a, b):
    """Starting solution."""
    c = 0.95 * b

    for _ in range(5):
        gamma = b / a
        kappa = c / b
        k2 = kappa ** 2
        gk2 = (gamma * kappa) ** 2
        fac1 = np.sqrt(1.0 - gk2)
        fac2 = np.sqrt(1.0 - k2)
        fr = (2.0 - k2) ** 2 - 4.0 * fac1 * fac2
        frp = -4.0 * (2.0 - k2) * kappa
        frp += 4.0 * fac2 * gamma * gamma * kappa / fac1
        frp += 4.0 * fac1 * kappa / fac2
        frp /= b
        c -= fr / frp

    return c

def getc(t, d, a, b, rho, mode, ifunc, dc):
    """Get phase velocity dispersion curve."""
    # Initialize arrays
    kmax = len(t)
    c = np.zeros(kmax, dtype=np.float64)
    cg = np.zeros(kmax, dtype=np.float64)

    # Preallocate Dunkin's matrix
    ca = np.empty((5, 5), dtype=np.float64)

    # Check for water layer
    llw = 0 if b[0] <= 0.0 else -1

    # Find the extremal velocities to assist in starting search
    betmx = -1.0e20
    betmn = 1.0e20
    mmax = len(d)
    for i in range(mmax):
        if b[i] > 0.01 and b[i] < betmn:
            betmn = b[i]
            jmn = i
            jsol = False
        elif b[i] < 0.01 and a[i] < betmn:
            betmn = a[i]
            jmn = i
            jsol = True
        betmx = max(betmx, b[i])

    # Solid layer solve halfspace period equation
    cc = betmn if jsol else gtsolh(a[jmn], b[jmn])

    # Back off a bit to get a starting value at a lower phase velocity
    cc *= 0.9
    c1 = cc
    cm = cc

    one = 1.0e-2
    onea = 1.5
    for iq in range(mode + 1):
        ibeg = 0
        iend = kmax

        del1st = 0.0
        for k in range(ibeg, iend):
            # Get initial phase velocity estimate to begin search
            ifirst = k == ibeg
            if ifirst and iq == 0:
                clow = cc
                c1 = cc
            elif ifirst and iq > 0:
                clow = c1
                c1 = c[ibeg] + one * dc
            elif not ifirst and iq > 0:
                clow = c[k] + one * dc
                c1 = max(c[k - 1], clow)
            elif not ifirst and iq == 0:
                clow = cm
                c1 = c[k - 1] - onea * dc

            # Bracket root and refine it
            c1, del1st, iret = getsol(
                t[k],
                c1,
                clow,
                dc,
                cm,
                betmx,
                ifirst,
                del1st,
                d,
                a,
                b,
                rho,
                ifunc,
                llw,
                ca,
            )

            if iret:
                if iq > 0:
                    for i in range(k, kmax):
                        cg[i] = 0.0

                    if iq == mode:
                        return cg
                    else:
                        c1 = 0.0
                        break
                else:
                    raise DispersionError("failed to find root for fundamental mode")

            c[k] = c1
            cg[k] = c[k]
            c1 = 0.0

    return cg
def dltar(wvno,omega,d,a,b,rho,ifunc,llw,ca):
    if ifunc!=2:raise ValueError('only Dunkin Rayleigh waves are selected')
    return dltar4(wvno,omega,d,a,b,rho,llw,ca)

def phase_velocity(period,thickness,velocity_p,velocity_s,density,dc=.0001):
    return getc(np.asarray(period),np.asarray(thickness),np.asarray(velocity_p),np.asarray(velocity_s),np.asarray(density),0,2,dc)

# Standalone evaluator: explicit forward algebra above, no solver imports.

import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
METRICS=tuple(f'velocity_{i}_absolute_error' for i in range(1,5))+tuple(f'thickness_{i}_absolute_error' for i in range(1,4))+('weighted_dispersion_rms',)

def read_output(path):
    with np.load(path,allow_pickle=False) as z:out={k:z[k].copy() for k in ('reconstruction','thickness')}
    for k,shape in [('reconstruction',(1,4)),('thickness',(1,3))]:
        a=out[k]
        if a.shape!=shape or a.dtype!=np.dtype('float64') or np.iscomplexobj(a) or not np.isfinite(a).all() or np.any(a<=0):raise ValueError(k+' must be positive finite real float64'+str(shape))
    return out

def dispersion_rms(velocity,thickness,period,observed,uncertainty):
    vs=velocity/1000.;nu=np.array([.3,.3,.49,.49]);vp=vs*np.sqrt((1-nu)/(.5-nu))
    calculated=phase_velocity(period,np.append(thickness/1000.,1.),vp,vs,np.full(4,1.95))
    if not np.isfinite(calculated).all() or np.any(calculated<=0):raise ValueError('invalid physical forward roots')
    return float(np.sqrt(np.mean(((calculated-observed)/uncertainty)**2)))

def evaluate(prediction):
    config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'));thresholds={}
    for key in METRICS:
        boundary=config[key+'_boundary']
        if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not np.isfinite(boundary) or boundary<0:raise ValueError('invalid metric boundary '+key)
        thresholds[key]={'comparison':'<=','boundary':boundary}
    estimate=read_output(prediction);reference=HERE/'ground_truth.npz'
    if not reference.exists():reference=HERE/'baseline_reference.npz'
    reference=read_output(reference);values={}
    for i,difference in enumerate(abs(estimate['reconstruction'][0]-reference['reconstruction'][0])):values[f'velocity_{i+1}_absolute_error']=float(difference)
    for i,difference in enumerate(abs(estimate['thickness'][0]-reference['thickness'][0])):values[f'thickness_{i+1}_absolute_error']=float(difference)
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:
        period=z['period'][0];observed=z['phase_velocity'][0];uncertainty=z['uncertainty'][0]
    values['weighted_dispersion_rms']=dispersion_rms(estimate['reconstruction'][0],estimate['thickness'][0],period,observed,uncertainty)
    if not np.isfinite(list(values.values())).all():raise ValueError('nonfinite computed metrics')
    passed=all(values[k]<=v['boundary'] for k,v in thresholds.items())
    return {'status':'pass' if passed else 'fail','passed':passed,'metrics':values,'thresholds':thresholds,'errors':[]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    try:report=evaluate(args.prediction);code=0 if report['passed'] else 1
    except Exception as exc:report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(exc)]};code=2
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
