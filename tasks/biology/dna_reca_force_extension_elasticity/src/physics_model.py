"""Explicit extensible Marko-Siggia elastic-chain equations and analytic cubic derivatives.

Copyright (c) 2025 LUMICKS B.V., Apache License 2.0 (see NOTICE).
Extracted only the named mathematical functions from the validated author source.
No file handling, fitting framework, or domain package is imported here.
Lengths: extension/contour micrometre, persistence nm; force/modulus pN.
"""
import numpy as np

def calc_cubic_root(a, b, c, selected_root):
    # Convert the equation to a depressed cubic for p and q, which'll allow us to greatly simplify the equations
    p = b - a * a / 3.0
    q = 2 * a * a * a / 27.0 - a * b / 3.0 + c
    det = q * q / 4.0 + p * p * p / 27.0

    # The model changes behaviours when the discriminant equates to zero. From this point we need a different root
    # resolution mechanism.
    sol = np.zeros(det.shape)
    mask = det >= 0

    sqrt_det = np.sqrt(det[mask])
    t1 = -q[mask] * 0.5 + sqrt_det
    t2 = -q[mask] * 0.5 - sqrt_det
    sol[mask] = np.cbrt(t1) + np.cbrt(t2)

    sqrt_minus_p = np.sqrt(-p[np.logical_not(mask)])
    q_masked = q[np.logical_not(mask)]

    asin_argument = 3.0 * np.sqrt(3.0) * q_masked / (2.0 * sqrt_minus_p**3)
    asin_argument = np.clip(asin_argument, -1.0, 1.0)

    if selected_root == 0:
        sol[np.logical_not(mask)] = (
            2.0 / np.sqrt(3.0) * sqrt_minus_p * np.sin((1.0 / 3.0) * np.arcsin(asin_argument))
        )
    elif selected_root == 1:
        sol[np.logical_not(mask)] = (
            -2.0
            / np.sqrt(3.0)
            * sqrt_minus_p
            * np.sin((1.0 / 3.0) * np.arcsin(asin_argument) + np.pi / 3.0)
        )
    elif selected_root == 2:
        sol[np.logical_not(mask)] = (
            2.0
            / np.sqrt(3.0)
            * sqrt_minus_p
            * np.cos((1.0 / 3.0) * np.arcsin(asin_argument) + np.pi / 6.0)
        )
    else:
        raise RuntimeError("Invalid root selected. Choose 0, 1 or 2.")

    return sol - a / 3.0


def calc_first_root(det, p, q, dp_da, dq_da, dq_db):
    # Calculate the first root for det < 0
    # Note that dp/dc = 0, dp_db = 1, dq_dc = 1

    sqrt_det = np.sqrt(det)
    term1 = np.abs(sqrt_det - 0.5 * q)  # Technically, the absolute is not part of this solution.
    term2 = np.abs(
        -sqrt_det - 0.5 * q
    )  # But it can cause numerical issues if we raise negative values to a root.
    t1 = term1 ** (2 / 3)
    t2 = term2 ** (2 / 3)

    # Derivatives of cube roots are not defined everywhere.
    #
    # The derivatives go over the bit where the cube roots are non-differentiable in the region of the model where it
    # switches from entropic to enthalpic behaviour. Even the finite difference derivatives look terrible here.
    #
    # When we approach Lc, t2 tends to zero, which causes problems with the division later.
    t1[abs(t1) < 1e-5] = 1e-5
    t2[abs(t2) < 1e-5] = 1e-5

    # When the discriminant goes to zero however, it means there are now repeated roots.
    sqrt_det[abs(sqrt_det) < 1e-5] = 1e-5

    # Compute all the elements required to evaluate the chain rule
    dy_ddet = 1.0 / (6.0 * sqrt_det * t1) - 1.0 / (6.0 * sqrt_det * t2)
    dy_dq = -1.0 / (6.0 * t1) - 1.0 / (6.0 * t2)
    dy_da = -(1.0 / 3.0)

    ddet_dp = p * p / 9.0
    ddet_dq = 0.5 * q

    # Total derivatives, terms multiplied by zero are omitted. Terms that are one are also omitted.
    # dp_db = dq_dc = 1
    total_dy_da = dy_ddet * ddet_dp * dp_da + dy_ddet * ddet_dq * dq_da + dy_dq * dq_da + dy_da
    total_dy_db = dy_ddet * ddet_dp + dy_ddet * ddet_dq * dq_db + dy_dq * dq_db
    total_dy_dc = dy_ddet * ddet_dq + dy_dq

    return total_dy_da, total_dy_db, total_dy_dc


def calc_triple_root(p, q, dp_da, dq_da, dq_db, root):
    # If we define:
    #   sqmp = sqrt(-p)
    #   F = 3 * sqrt(3) * q / (2 * sqmp**3 )
    #
    # Then the solution is:
    #   2 /sqrt(3) * sqmp * cos((1/3) * asin(F) + pi/6) - a / 3
    #
    # Note that dp/dc = 0, dp_db = 1, dq_dc = 1
    # The rest of this file is simply applying the chain rule.
    sqmp = np.sqrt(-p)
    F = 3.0 * np.sqrt(3.0) * q / (2.0 * sqmp**3)

    dF_dsqmp = -9 * np.sqrt(3) * q / (2 * sqmp**4)
    dF_dq = 3.0 * np.sqrt(3) / (2.0 * sqmp**3)
    dsqmp_dp = -1.0 / (2.0 * sqmp)
    dy_da = -1.0 / 3.0

    if root == 0:
        arg = np.arcsin(F) / 3.0
        dy_dsqmp = 2.0 * np.sqrt(3.0) * np.sin(arg) / 3.0
        dy_dF = 2.0 * np.sqrt(3.0) * sqmp * np.cos(arg) / (9.0 * np.sqrt(1.0 - F**2))
    elif root == 1:
        arg = np.arcsin(F) / 3.0 + np.pi / 3.0
        dy_dsqmp = -2.0 * np.sqrt(3.0) * np.sin(arg) / 3.0
        dy_dF = -2.0 * np.sqrt(3.0) * sqmp * np.cos(arg) / (9.0 * np.sqrt(1.0 - F**2))
    elif root == 2:
        arg = np.arcsin(F) / 3.0 + np.pi / 6.0
        dy_dsqmp = 2.0 * np.sqrt(3.0) * np.cos(arg) / 3.0
        dy_dF = -2.0 * np.sqrt(3.0) * sqmp * np.sin(arg) / (9.0 * np.sqrt(1.0 - F * F))

    # Total derivatives
    total_dy_da = (
        dy_dsqmp * dsqmp_dp * dp_da + dy_dF * (dF_dsqmp * dsqmp_dp * dp_da + dF_dq * dq_da) + dy_da
    )
    total_dy_db = dy_dsqmp * dsqmp_dp + dy_dF * (dF_dsqmp * dsqmp_dp + dF_dq * dq_db)
    total_dy_dc = dy_dF * dF_dq

    return total_dy_da, total_dy_db, total_dy_dc


def calc_cubic_root_derivatives(a, b, c, selected_root):
    """Calculate the root derivatives of a cubic polynomial with respect to the polynomial coefficients.

    For a polynomial of the form:
        x**3 + a * x**2 + b * x + c = 0

    Note that this is not a general root-finding function, but tailored for use with the inverted WLC model. For det < 0
    it returns the derivatives of the first root. For det > 0 it returns the derivatives of the third root.

    Parameters
    ----------
    a, b, c: array_like
        Coefficients of the reduced cubic polynomial.
        x**3 + a x**2 + b x + c = 0
    selected_root: integer
        which root to compute the derivative of
    """
    p = b - a * a / 3.0
    q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c
    det = q * q / 4.0 + p * p * p / 27.0

    # Determine derivatives of our transformation from polynomial coefficients to helper coordinates p and q
    dp_da = -2.0 * a / 3.0
    dq_da = 2.0 * a**2.0 / 9.0 - b / 3.0
    dq_db = -a / 3.0

    total_dy_da = np.zeros(det.shape)
    total_dy_db = np.zeros(det.shape)
    total_dy_dc = np.zeros(det.shape)

    mask = det > 0
    total_dy_da[mask], total_dy_db[mask], total_dy_dc[mask] = calc_first_root(
        det[mask], p[mask], q[mask], dp_da[mask], dq_da[mask], dq_db[mask]
    )

    nmask = np.logical_not(mask)
    (total_dy_da[nmask], total_dy_db[nmask], total_dy_dc[nmask]) = calc_triple_root(
        p[nmask], q[nmask], dp_da[nmask], dq_da[nmask], dq_db[nmask], selected_root
    )

    return total_dy_da, total_dy_db, total_dy_dc


def ewlc_marko_siggia_force(d, Lp, Lc, St, kT=4.11):
    """Marko Siggia's Worm-like Chain model with force as the dependent variable

    Modified Marko Siggia's Worm-like Chain model. Modification of Marko-Siggia formula [1]_
    to incorporate enthalpic stretching. Has limitations similar to Marko-Siggia
    near F = 0.1 pN [2]_.

    Parameters
    ----------
    d : array_like
        extension [um]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    St : float
        stretching modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]

    References
    ----------
    .. [1] J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26, 8759-8770 (1995).
    .. [2] Wang, M. D., Yin, H., Landick, R., Gelles, J., & Block, S. M. (1997). Stretching DNA
           with optical tweezers. Biophysical journal, 72(3), 1335-1346.
    """
    if Lp <= 0 or Lc <= 0 or St <= 0 or kT <= 0:
        raise ValueError(
            "Persistence length, contour length, stretch modulus and kT must be bigger than 0"
        )

    c = -(St**3) * d * kT * (1.5 * Lc**2 - 2.25 * Lc * d + d**2) / (Lc**3 * (Lp * St + kT))
    b = (
        St**2
        * (
            Lc**2 * Lp * St
            + 1.5 * Lc**2 * kT
            - 2 * Lc * Lp * St * d
            - 4.5 * Lc * d * kT
            + Lp * St * d**2
            + 3 * d**2 * kT
        )
        / (Lc**2 * (Lp * St + kT))
    )
    a = (
        St
        * (2 * Lc * Lp * St + 2.25 * Lc * kT - 2 * Lp * St * d - 3 * d * kT)
        / (Lc * (Lp * St + kT))
    )

    return calc_cubic_root(a, b, c, 2)


def ewlc_marko_siggia_force_jac(d, Lp, Lc, St, kT=4.11):
    c = -(St**3) * d * kT * (1.5 * Lc**2 - 2.25 * Lc * d + d**2) / (Lc**3 * (Lp * St + kT))
    b = (
        St**2
        * (
            Lc**2 * Lp * St
            + 1.5 * Lc**2 * kT
            - 2 * Lc * Lp * St * d
            - 4.5 * Lc * d * kT
            + Lp * St * d**2
            + 3 * d**2 * kT
        )
        / (Lc**2 * (Lp * St + kT))
    )
    a = (
        St
        * (2 * Lc * Lp * St + 2.25 * Lc * kT - 2 * Lp * St * d - 3 * d * kT)
        / (Lc * (Lp * St + kT))
    )

    total_dy_da, total_dy_db, total_dy_dc = calc_cubic_root_derivatives(a, b, c, 2)

    # Map back to our output parameters
    denom1 = Lc**3 * (Lp * St + kT) ** 2
    denom2 = Lc * (Lp**2 * St**2 + 2.0 * Lp * St * kT + kT**2)
    dkT = d * kT
    dc_dLc = St**3 * dkT * (1.5 * Lc**2 - 4.5 * Lc * d + 3.0 * d**2) / (Lc**4 * (Lp * St + kT))
    dc_dSt = -(St**2) * dkT * (2 * Lp * St + 3 * kT) * (1.5 * Lc**2 - 2.25 * Lc * d + d**2) / denom1
    dc_dLp = St**4 * dkT * (1.5 * Lc**2 - 2.25 * Lc * d + d**2) / denom1
    dc_dkT = -Lp * St**4 * d * (1.5 * Lc**2 - 2.25 * Lc * d + d**2) / denom1
    db_dLc = (
        St**2
        * d
        * (2.0 * Lc * Lp * St + 4.5 * Lc * kT - 2.0 * Lp * St * d - 6.0 * d * kT)
        / (Lc**3 * (Lp * St + kT))
    )
    db_dSt = (
        St
        * (
            2.0 * Lc**2 * Lp**2 * St**2
            + 4.5 * Lc**2 * Lp * St * kT
            + 3.0 * Lc**2 * kT**2
            - 4.0 * Lc * Lp**2 * St**2 * d
            - 10.5 * Lc * Lp * St * d * kT
            - 9.0 * Lc * d * kT**2
            + 2.0 * Lp**2 * St**2 * d**2
            + 6.0 * Lp * St * d**2 * kT
            + 6.0 * d**2 * kT**2
        )
        / (Lc * denom2)
    )
    db_dLp = -(St**3) * kT * (0.5 * Lc**2 - 2.5 * Lc * d + 2.0 * d**2) / (Lc * denom2)
    db_dkT = Lp * St**3 * (0.5 * Lc**2 - 2.5 * Lc * d + 2.0 * d**2) / (Lc * denom2)
    da_dLc = St * d * (2 * Lp * St + 3 * kT) / (Lc**2 * (Lp * St + kT))
    da_dSt = (
        -Lp * St * (2 * Lc * Lp * St + 2.25 * Lc * kT - 2 * Lp * St * d - 3 * d * kT)
        + (Lp * St + kT)
        * (
            2 * Lc * Lp * St
            + 2.25 * Lc * kT
            - 2 * Lp * St * d
            + 2 * Lp * St * (Lc - d)
            - 3 * d * kT
        )
    ) / (Lc * (Lp * St + kT) ** 2)
    da_dLp = -(St**2) * kT * (0.25 * Lc - d) / denom2
    da_dkT = Lp * St**2 * (0.25 * Lc - d) / denom2

    # Terms multiplied by zero are omitted. Terms that are one are also omitted.
    total_dy_dLp = total_dy_da * da_dLp + total_dy_db * db_dLp + total_dy_dc * dc_dLp
    total_dy_dLc = total_dy_da * da_dLc + total_dy_db * db_dLc + total_dy_dc * dc_dLc
    total_dy_dSt = total_dy_da * da_dSt + total_dy_db * db_dSt + total_dy_dc * dc_dSt
    total_dy_dkT = total_dy_da * da_dkT + total_dy_db * db_dkT + total_dy_dc * dc_dkT

    return [total_dy_dLp, total_dy_dLc, total_dy_dSt, total_dy_dkT]


def ewlc_marko_siggia_force_derivative(d, Lp, Lc, St, kT=4.11):
    c = -(St**3) * d * kT * (1.5 * Lc**2 - 2.25 * Lc * d + d**2) / (Lc**3 * (Lp * St + kT))
    b = (
        St**2
        * (
            Lc**2 * Lp * St
            + 1.5 * Lc**2 * kT
            - 2 * Lc * Lp * St * d
            - 4.5 * Lc * d * kT
            + Lp * St * d**2
            + 3 * d**2 * kT
        )
        / (Lc**2 * (Lp * St + kT))
    )
    a = (
        St
        * (2 * Lc * Lp * St + 2.25 * Lc * kT - 2 * Lp * St * d - 3 * d * kT)
        / (Lc * (Lp * St + kT))
    )

    total_dy_da, total_dy_db, total_dy_dc = calc_cubic_root_derivatives(a, b, c, 2)

    # Map back to our output parameters
    denom = Lc * (Lp * St + kT)
    dc_dd = -(St**3) * kT * (1.5 * Lc**2 - 4.5 * Lc * d + 3.0 * d**2) / (Lc * Lc * denom)
    db_dd = (
        St**2 * (-2 * Lc * Lp * St - 4.5 * Lc * kT + 2 * Lp * St * d + 6 * d * kT) / (Lc * denom)
    )
    da_dd = -St * (2 * Lp * St + 3 * kT) / denom

    return total_dy_da * da_dd + total_dy_db * db_dd + total_dy_dc * dc_dd

