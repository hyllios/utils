"""
The special functions of the Sanna-Pellegrini-Gross SCDFT functional
(A. Sanna, C. Pellegrini, E. K. U. Gross, Phys. Rev. Lett. 125, 057001 (2020)),
evaluated so they survive the temperatures and energy ranges of a real
calculation.

Both are written in the paper in forms that cannot be used literally:

`I` (their Eq. 12) carries factors exp(beta*xi) with xi up to electronvolts and
beta up to 1e4 Ha^-1, so exp overflows long before any physics is reached, and
it has two removable singularities where its denominators vanish. Multiplying
the Fermi and Bose prefactors in, using f(x) exp(beta x) = 1 - f(x) and
n(w) exp(beta w) = 1 + n(w), gives the bounded equivalent

    I = [(1-f)f' n - f(1-f')(1+n)] / (xi - xi' - w)
      - [f(1-f')n - (1-f)f'(1+n)] / (xi - xi' + w)

and, since the two numerator terms differ exactly by exp(beta d), the small-d
limit follows from expm1 without cancellation.

`J` (their Eq. 13) is, term for term, the second-order NEWTON DIVIDED DIFFERENCE
of the Fermi function over the three nodes {xi - w, E, gamma}:

    J(xi, E, w, gamma) = [f(xi) + n(w)] * f[xi - w, E, gamma]

which is why its three apparent poles -- at xi-w = E, xi-w = gamma and
gamma = E -- are all removable: a divided difference is finite when its nodes
coalesce, tending to a derivative. Computing it as a divided difference gets
that right by construction instead of relying on cancellation between large
terms, which matters because the s1 s2 s3 sum in their Eq. (11) sweeps E and
gamma across each other.

Verified against the literal formulas to 10 digits wherever those can be
evaluated at all (tests/test_utils_scdft.py).

Atomic units throughout: energies in Hartree, beta in 1/Hartree.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fermi", "bose", "i_function", "di_dxi", "j_function", "j_lueders",
           "p_smooth", "i_akashi", "j_akashi"]

#: |beta*d| below which the expm1 form of I is used instead of the difference
_SMALL = 1.0
#: |beta*dx| below which a divided difference is replaced by a derivative
_DD_SMALL = 1e-6


def fermi(xi, beta: float):
    """Fermi function, written through tanh so it saturates instead of
    overflowing for large |beta*xi|."""
    return 0.5 * (1.0 - np.tanh(0.5 * beta * np.asarray(xi, dtype=np.float64)))


def bose(omega, beta: float):
    """
    Bose function 1/(exp(beta*w) - 1), via expm1.

    Negative arguments are allowed and give n(-w) = -(1 + n(w)): the s2 = -1
    branch of the paper's Eq. (11) needs exactly that.
    """
    with np.errstate(over="ignore"):
        # expm1 overflows for beta*w > 709, where n(w) is 0 to machine
        # precision anyway -- 1/inf gives exactly that
        return 1.0 / np.expm1(beta * np.asarray(omega, dtype=np.float64))


def _g_expm1(x):
    """expm1(x)/x, continued to 1 at x = 0."""
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    nz = x != 0.0
    out[nz] = np.expm1(x[nz]) / x[nz]
    return out


def i_function(xi, xi_p, omega, beta: float):
    """
    Their Eq. (12), broadcast over its arguments.

    Args:
      xi, xi_p : electron energies relative to E_F, Hartree.
      omega    : phonon frequency, Hartree, strictly positive.
      beta     : 1/kT in 1/Hartree.
    """
    xi, xi_p, omega = (np.asarray(v, dtype=np.float64) for v in (xi, xi_p, omega))
    f, fp = fermi(xi, beta), fermi(xi_p, beta)
    n = bose(omega, beta)
    d1 = xi - xi_p - omega
    d2 = xi - xi_p + omega

    # Away from the singularities the plain difference is well conditioned: the
    # two numerator terms differ by a factor exp(beta*d), far from 1.
    n1 = (1.0 - f) * fp * n - f * (1.0 - fp) * (1.0 + n)
    n2 = f * (1.0 - fp) * n - (1.0 - f) * fp * (1.0 + n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(d1 != 0.0, n1 / np.where(d1 != 0.0, d1, 1.0), 0.0)
        t2 = np.where(d2 != 0.0, n2 / np.where(d2 != 0.0, d2, 1.0), 0.0)

    # Near them, use the exact rewriting n1 = f(1-f')(1+n)*expm1(beta*d1),
    # whose expm1(x)/x is smooth through zero.
    small1 = np.abs(beta * d1) < _SMALL
    small2 = np.abs(beta * d2) < _SMALL
    # mask the arguments before calling expm1: it overflows for the large-|d|
    # entries, and np.where would still evaluate them (inf * 0 -> nan)
    arg1 = np.where(small1, beta * d1, 0.0)
    arg2 = np.where(small2, -beta * d2, 0.0)
    if np.any(small1):
        t1 = np.where(small1,
                      beta * f * (1.0 - fp) * (1.0 + n) * _g_expm1(arg1), t1)
    if np.any(small2):
        # n2 = (1-f) f' (1+n) expm1(-beta*d2), and expm1(-x)/x = -G(-x), so this
        # branch carries a minus sign the d1 branch does not.
        t2 = np.where(small2,
                      -beta * (1.0 - f) * fp * (1.0 + n) * _g_expm1(arg2), t2)
    return t1 - t2


def _g_expm1_prime(x):
    """d/dx [expm1(x)/x] = (x e^x - expm1(x))/x^2, continued to 1/2 at x = 0."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, 0.5)
    big = np.abs(x) > 1e-5
    if np.any(big):
        xb = x[big]
        out[big] = (xb * np.exp(xb) - np.expm1(xb)) / xb ** 2
    small = ~big & (x != 0.0)
    if np.any(small):                      # series: 1/2 + x/3 + x^2/8
        xs = x[small]
        out[small] = 0.5 + xs / 3.0 + xs ** 2 / 8.0
    return out


def di_dxi(xi, xi_p, omega, beta: float):
    """
    d/dxi of `i_function`, in closed form.

    Eq. (4) of the kernels needs the xi-derivative of
    I(xi, xi', w) + I(xi, -xi', w). Differentiating analytically rather than by
    finite difference removes the only approximation this module made to a
    published closed form. The same two-branch treatment as `i_function` is
    required: the direct quotient loses precision as a denominator vanishes, and
    the expm1 form overflows once |beta*d| is large.

    With df/dxi = -beta f (1-f), the direct branch differentiates N/d as
    (dN/dxi)/d - N/d^2, and the small-|d| branch differentiates
    beta f (1-f') (1+n) G(beta d) with G(x) = expm1(x)/x, needing G'.
    """
    xi, xi_p, omega = (np.asarray(v, dtype=np.float64) for v in (xi, xi_p, omega))
    f, fp = fermi(xi, beta), fermi(xi_p, beta)
    n = bose(omega, beta)
    df = -beta * f * (1.0 - f)
    d1 = xi - xi_p - omega
    d2 = xi - xi_p + omega

    n1 = (1.0 - f) * fp * n - f * (1.0 - fp) * (1.0 + n)
    n2 = f * (1.0 - fp) * n - (1.0 - f) * fp * (1.0 + n)
    dn1 = -df * (fp * n + (1.0 - fp) * (1.0 + n))
    dn2 = df * ((1.0 - fp) * n + fp * (1.0 + n))
    with np.errstate(divide="ignore", invalid="ignore"):
        safe1 = np.where(d1 != 0.0, d1, 1.0)
        safe2 = np.where(d2 != 0.0, d2, 1.0)
        t1 = np.where(d1 != 0.0, dn1 / safe1 - n1 / safe1 ** 2, 0.0)
        t2 = np.where(d2 != 0.0, dn2 / safe2 - n2 / safe2 ** 2, 0.0)

    small1 = np.abs(beta * d1) < _SMALL
    small2 = np.abs(beta * d2) < _SMALL
    arg1 = np.where(small1, beta * d1, 0.0)
    arg2 = np.where(small2, -beta * d2, 0.0)
    if np.any(small1):
        # t1 = beta f (1-f') (1+n) G(beta d1)
        alt = beta * (1.0 + n) * (1.0 - fp) * (
            df * _g_expm1(arg1) + f * beta * _g_expm1_prime(arg1))
        t1 = np.where(small1, alt, t1)
    if np.any(small2):
        # t2 = -beta (1-f) f' (1+n) G(-beta d2), and d(-beta d2)/dxi = -beta
        alt = -beta * (1.0 + n) * fp * (
            -df * _g_expm1(arg2) - beta * (1.0 - f) * _g_expm1_prime(arg2))
        t2 = np.where(small2, alt, t2)
    return t1 - t2


def _f_prime(x, beta: float):
    """df/dx = -(beta/4) sech^2(beta x/2)."""
    with np.errstate(over="ignore"):
        # cosh^2 overflows deep in the tails, where f' is 0 to machine precision
        return -0.25 * beta / np.cosh(0.5 * beta * np.asarray(x, np.float64)) ** 2


def _f_second(x, beta: float):
    """d2f/dx2 = (beta^2/4) sech^2(beta x/2) tanh(beta x/2)."""
    h = 0.5 * beta * np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore"):
        return 0.25 * beta ** 2 * np.tanh(h) / np.cosh(h) ** 2


def _dd1(x, y, beta: float):
    """First divided difference f[x, y], -> f'(x) as y -> x."""
    dx = x - y
    close = np.abs(beta * dx) < _DD_SMALL
    safe = np.where(close, 1.0, dx)
    out = (fermi(x, beta) - fermi(y, beta)) / safe
    return np.where(close, _f_prime(x, beta), out)


def _dd2(a, b, c, beta: float):
    """
    Second divided difference f[a, b, c], stable as nodes coalesce.

    Divided differences are symmetric in their nodes, so the three are sorted
    and the OUTER difference is taken across the widest pair. That single
    choice keeps the outer denominator as large as it can be, which is what
    makes the confluent cases harmless.
    """
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    mid = a + b + c - lo - hi
    span = lo - hi
    close = np.abs(beta * span) < _DD_SMALL
    safe = np.where(close, 1.0, span)
    out = (_dd1(lo, mid, beta) - _dd1(mid, hi, beta)) / safe
    return np.where(close, 0.5 * _f_second(lo, beta), out)


def j_function(xi, energy, omega, gamma, beta: float):
    """
    Their Eq. (13), as ``[f(xi) + n(omega)]`` times the second divided
    difference of f over ``{xi - omega, energy, gamma}``.

    Args:
      xi     : electron energy relative to E_F, Hartree.
      energy : the +-sqrt(xi'^2 + gamma3 Delta'^2) of their Eq. (11).
      omega  : phonon frequency (may be negative: the s2 = -1 branch).
      gamma  : the +-gamma2*omega of their Eq. (11).
      beta   : 1/kT in 1/Hartree.
    """
    xi, energy, omega, gamma = (np.asarray(v, dtype=np.float64)
                                for v in (xi, energy, omega, gamma))
    pref = fermi(xi, beta) + bose(omega, beta)
    return pref * _dd2(xi - omega, energy, gamma, beta)


def j_lueders(xi, xi_p, omega, beta: float):
    """
    The J of Luders et al., Phys. Rev. B 72, 024545 (2005), their Eqs. (80)-(81),
    used by the LM2005 renormalisation kernel Z^ph (their Eq. 79).

        J(xi, xi', W)  = Jt(xi, xi', W) - Jt(xi, xi', -W)
        Jt(xi, xi', W) = -[f(xi) + n(W)]/(xi-xi'-W)
                          * [ (f(xi') - f(xi-W))/(xi-xi'-W)
                              - beta f(xi-W) f(-xi+W) ]

    Like `j_function`, this is a second divided difference in disguise, here with
    a REPEATED node. Writing a = xi - W and using -beta f(a) f(-a) = f'(a),

        Jt = -[f(xi) + n(W)] * ( f'(a) - f[a, xi'] ) / (a - xi')
           = -[f(xi) + n(W)] * f[a, a, xi'] ,

    so the apparent double pole at xi' = xi - W is removable, and evaluating it
    as `_dd2` with a doubled node is stable by construction rather than relying
    on cancellation between two large terms.

    A note on the source: the scanned Eq. (81) reads ``f(-xi' + W)`` in the
    second bracketed term, but that cannot be right -- the 1/(xi-xi'-W)^2 pole
    only cancels if the argument is ``-xi + W``, and Akashi and Arita
    (Phys. Rev. B 88, 014514 (2013), their Eq. 44) reproduce it as ``f(-xi + W)``.
    The latter is used here.

    Args:
      xi, xi_p : electron energies relative to E_F, Hartree.
      omega    : phonon frequency, Hartree, strictly positive.
      beta     : 1/kT in 1/Hartree.
    """
    xi, xi_p, omega = (np.asarray(v, dtype=np.float64) for v in (xi, xi_p, omega))

    def _jt(w):
        a = xi - w
        return -(fermi(xi, beta) + bose(w, beta)) * _dd2(a, a, xi_p, beta)

    return _jt(omega) - _jt(-omega)


# ---------------------------------------------------------------------------
# Akashi and Arita, Phys. Rev. B 88, 014514 (2013) [arXiv:1305.0390]:
# the renormalisation kernel WITHOUT particle-hole symmetrisation.
#
# Luders et al. symmetrised Z^ph in xi because the unsymmetrised form diverges,
# which throws away the antisymmetric part of the DOS. Akashi and Arita show the
# divergences cancel analytically between their I and 2J, so no symmetrisation
# is needed and N(xi')/N(0) can carry a genuine asymmetry.
#
# Notation below follows the paper: a = xi - omega, s = xi' + omega.
# ---------------------------------------------------------------------------

#: sharpness of the smoothing p(x); the paper's value, and it states the result
#: is insensitive to the choice
_P_SHARPNESS = 500.0


def p_smooth(x, beta: float, sharpness: float = _P_SHARPNESS):
    """
    Their even smoothing function ``p(x) = [tanh(sharpness*beta*x)]^4``.

    Unity for ``|x| >~ T`` and O(x^2) (in fact O(x^4)) for ``|x| << T``, which
    is what lets ``p(s)/s`` stand in for the principal value ``P[1/s]``.
    """
    return np.tanh(sharpness * beta * np.asarray(x, dtype=np.float64)) ** 4


def _p_over_x(x, beta: float, sharpness: float = _P_SHARPNESS):
    """``p(x)/x``, taken to its limit 0 at x = 0.

    For small argument ``tanh(c x)^4 / x -> c^4 x^3``, so the branch is not a
    fudge: it is the leading term of the same expression.
    """
    x = np.asarray(x, dtype=np.float64)
    c = sharpness * beta
    small = np.abs(c * x) < 1e-3
    safe = np.where(small, 1.0, x)
    out = np.tanh(c * safe) ** 4 / safe
    return np.where(small, c ** 4 * x ** 3, out)


def _i_tilde_akashi(xi, xi_p, omega, beta: float,
                    sharpness: float = _P_SHARPNESS):
    """
    Their Eq. (42),

        It(xi, xi', w) = [f(xi) + n(w)] (f(xi') - f(xi-w))/(xi-xi'-w)
                         * p(xi'+w)/(xi'+w)

    written through a divided difference. With a = xi - w the middle factor is
        (f(xi') - f(a)) / (a - xi') = -f[a, xi'] ,
    so
        It = -[f(xi) + n(w)] f[a, xi'] p(s)/s ,
    which is smooth at xi' = a (where the quotient becomes f'(a)) as well as at
    s = 0. Nothing is cancelled numerically.
    """
    xi, xi_p, omega = (np.asarray(v, dtype=np.float64)
                       for v in (xi, xi_p, omega))
    a = xi - omega
    s = xi_p + omega
    pref = fermi(xi, beta) + bose(omega, beta)
    return -pref * _dd1(a, xi_p, beta) * _p_over_x(s, beta, sharpness)


def i_akashi(xi, xi_p, omega, beta: float, sharpness: float = _P_SHARPNESS):
    """
    Their Eq. (41): ``I = It(w) - It(-w) - It(-xi') + It(-xi',-w)``.

    NOTE this is ODD in xi' by construction, so for a particle-hole SYMMETRIC
    DOS its xi' integral vanishes identically and the whole I term drops out of
    Z. It contributes only through the antisymmetric part of N(xi'), which is
    precisely the physics the paper adds.
    """
    def it(xp, w):
        return _i_tilde_akashi(xi, xp, w, beta, sharpness)

    return (it(xi_p, omega) - it(xi_p, -omega)
            - it(-np.asarray(xi_p, dtype=np.float64), omega)
            + it(-np.asarray(xi_p, dtype=np.float64), -omega))


def _j_tilde_akashi(xi, xi_p, omega, beta: float,
                    sharpness: float = _P_SHARPNESS):
    """
    Their Eq. (44),

        Jt = -[f(xi)+n(w)]/(xi-xi'-w) p(s)
              * [ (f(xi')-f(xi-w))/(xi-xi'-w)
                  - beta f(xi-w) f(-xi+w) xi/s ]

    with the 1/(xi-xi'-w) prefactor resolved ANALYTICALLY rather than left to
    cancel numerically. Writing a = xi - w, s = xi' + w, d = a - xi' = xi - s,
    and using f'(a) = -beta f(a) f(-a), the bracket is

        B = -f[a,xi'] + f'(a) xi/s .

    At xi' = a one has s = xi exactly, so xi/s = 1 and B = f'(a) - f'(a) = 0:
    the apparent double pole is removable, which is the cancellation the paper
    proves. Expanding about it,

        B = d ( f[a,a,xi'] + f'(a)/s )   =>   Jt = -[f+n] p(s) B/d
                                              = -[f+n] p(s) ( f[a,a,xi'] + f'(a)/s ) ,

    exactly. The second term is evaluated as f'(a) p(s)/s so that it, too, is
    smooth through s = 0.

    Sanity relation to the older kernel: with p -> 1 and the xi/s factor set to
    1 this reduces to `j_lueders`'s Jt = -[f+n] f[a,a,xi'].
    """
    xi, xi_p, omega = (np.asarray(v, dtype=np.float64)
                       for v in (xi, xi_p, omega))
    a = xi - omega
    s = xi_p + omega
    pref = fermi(xi, beta) + bose(omega, beta)
    return -pref * (p_smooth(s, beta, sharpness) * _dd2(a, a, xi_p, beta)
                    + _f_prime(a, beta) * _p_over_x(s, beta, sharpness))


def j_akashi(xi, xi_p, omega, beta: float, sharpness: float = _P_SHARPNESS):
    """
    Their Eq. (43): ``J = Jt(w) - Jt(-w)``.

    Unlike `i_akashi` this is NOT antisymmetrised in xi'; the xi' -> -xi'
    combination appears only where Z assembles ``I - 2J``.
    """
    return (_j_tilde_akashi(xi, xi_p, omega, beta, sharpness)
            - _j_tilde_akashi(xi, xi_p, -np.asarray(omega, dtype=np.float64),
                              beta, sharpness))
