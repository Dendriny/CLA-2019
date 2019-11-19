import numpy as np


def _sign(x):
    return 1 if np.real(x) >= 0 else -1  # eq. 2


def _householder(a):
    x = a[:, 0].reshape(-1, 1)
    e1 = np.zeros_like(x)
    e1[0] = 1.
    v = _sign(x[0]) * norm(x) * e1 + x
    v /= norm(v)
    if np.iscomplexobj(x):
        h = (1. + np.vdot(v, x).conj() / np.vdot(v, x)) * v @ v.T.conj()  # eq. 1
    else:
        h = 2 * v @ v.T
    return h


def norm(x):
    """
    2-norm of a matrix or vector.
    :param x: array, Matrix for which norm is calculated
    :return: 2-norm of x
    """
    return np.sqrt(np.sum(np.square(np.real(x)) + np.square(np.imag(x))))


def qr(a, b=None, reduced=True, inplace=False):
    """
    qr factorization of a matrix by Householder reflections.
    q is an orthogonal/unitary matrix and r is upper-triangular.
    :param a: array, shape (m, n); Matrix to be factored
    :param b: if not None updates the vector b in Ax=b system of equations being solved
    :param reduced: if True returns q, r with dimensions (m, k), (k, n)
    :param inplace: if True directly updates the matrix a. Does not return r
    :return: q, r (if not inplace), b (optional)
    """
    mn = min(a.shape)
    m = a.shape[0]
    q = np.identity(m).astype(a.dtype)

    if not inplace:
        a = a.copy()

    h = np.zeros(q.shape).astype(a.dtype)

    for k in range(mn):
        h[k:, k:] = _householder(a[k:, k:])
        a[k:, k:] -= h[k:, k:] @ a[k:, k:]
        qk = np.identity(m).astype(a.dtype)
        qk[k:, k:] -= h[k:, k:]
        q = qk @ q

        try:
            b[k:] -= h[k:, k:] @ b[k:]
        except TypeError:
            continue

    q = q.T.conj()
    if reduced:
        q = q[:, :mn]
        a = a[:mn]

    if b is not None and not inplace:
        return q, a, b
    elif b is None and not inplace:
        return q, a
    elif b is not None:
        return q, b
    else:
        return q


def mgs(a):
    """
    qr factorization of a matrix by Modified Gram-Schmidt method.
    q is an orthogonal/unitary matrix and r is upper-triangular.
    :param a: array, shape (m, n); Matrix to be factored
    :return: q, r
    """
    mn = min(a.shape)
    m, n = a.shape
    q = np.identity(m).astype(a.dtype)
    v = np.zeros_like(a)
    r = np.zeros_like(a)
    for i in range(n):
        v[:, i] = a[:, i]
    for i in range(mn):
        r[i, i] = norm(v[:, i])
        q[:, i] = v[:, i] / r[i, i]
        for j in range(1, n):
            r[i, j] = np.vdot(q[:, i], v[:, j])
            v[:, j] -= r[i, j] * q[:, i]

    return q[:, :mn], r[:mn]

def solve(a, b):
    """
    Solve a linear matrix equation ax = b for x.
    :param a: matrix of independent variables, shape (m, n)
    :param b: dependent variable values
    :return: solution x to the system ax = b
    """
    _, r, b = qr(a, b)
    return solve_triu(r, b)


def solve_triu(r, b):
    """
    Solve a linear matrix equation ax = b for x where a is upper triangular.
    :param r: upper triangular matrix of independent variables
    :param b: dependent variable values
    :return: x: solution to the system rx = b
    """
    assert np.allclose(r, np.triu(r))
    x = np.zeros((r.shape[1], 1)).astype(r.dtype)
    x[-1] = b[-1] / r[-1, -1]
    for j in range(r.shape[0] - 2, -1, -1):
        x[j] = (b[j] - r[j, j + 1:] @ x[j + 1:]) / r[j, j]
    return x


def solve_tril(l, b):
    """
    Solve a linear matrix equation ax = b for x where a is upper triangular.
    :param l: lower triangular matrix of independent variables
    :param b: dependent variable values
    :return: x: solution to the system lx = b
    """
    assert np.allclose(l, np.tril(l))
    x = np.zeros((l.shape[1], 1)).astype(l.dtype)
    x[0] = b[0] / l[0, 0]
    for j in range(1, l.shape[0], 1):
        x[j] = (b[j] - l[j, :j] @ x[:j]) / l[j, j]
    return x


def solve_diag(d, b):
    """
    Solve a linear matrix equation ax = b for x where a is diagonal.
    :param d: diagonal matrix of independent variables
    :param b: dependent variable values
    :return: x: solution to the system dx = b
    """
    assert np.allclose(d, np.diag(np.diag(d)))
    x = b / np.diag(d).reshape(b.shape)
    return x


def poly1d(c, x):
    """
    Caculates y = Xc where X is a Vandermonde matrix of coefficients x.
    :param c: array, polynomial coefficients
    :param x: array, independent variables for which to calculate y
    :return:
    """
    return np.column_stack([x**i for i in range(len(c))]) @ c


def polyfit(x, y, deg):
    """
    Fits a polynomial p(x) = c[deg] * x**deg + ... + c[0] of degree 'deg'
    to points (x, y) by minimising the sum of squared residuals.
    :param x: array, x-coordinates of observations
    :param y: array, y-coordinates of observations
    :param deg: int, degree of the fitting polynomial
    :return: c: array of polynomial coefficients, lowest power first
             residuals: residuals of the least-squares fit
    """
    vander = np.column_stack([x**i for i in range(deg + 1)])
    # scale to improve condition number
    scale = np.sqrt((vander*vander).sum(axis=0))
    vander /= scale
    q, r = qr(vander)
    b = q.T.conj() @ y
    c = solve_triu(r, b)
    residuals = norm(y - c.reshape(-1) @ vander.T.conj()) ** 2
    c = (c.T/scale).T  # rescale

    return c, residuals
