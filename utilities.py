import numpy as np


def norm(x):
    # return np.sqrt(np.sum(np.square(np.real(x)) + np.square(np.imag(x))))
    return np.sqrt(x.T.conj() @ x)


def sign(x):
    return 1 if np.real(x) >= 0 else -1


def householder_ab(a, b):
    mn = min(a.shape)
    for k in range(mn):
        x = a[k:, k].reshape(-1, 1)
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = sign(x[0]) * norm(x) * e1 + x
        v /= norm(v)
        a[k:, k:] -= 2 * v @ v.T.conj() @ a[k:, k:]

        b[k:] -= 2 * v @ v.T.conj() @ b[k:]
    return np.triu(a), b


def qr(a: np.ndarray, reduced=True, inplace=True):
    mn = min(a.shape)
    m, _ = a.shape
    q = np.identity(m).astype(a.dtype)
    if not inplace:
        a = a.copy()

    for k in range(mn):
        x = a[k:, k].reshape(-1, 1)
        e1 = np.zeros_like(x)
        e1[0] = 1.
        v = sign(x[0]) * norm(x) * e1 + x
        v /= norm(v)
        H = (1. + (v.T.conj() @ x).conj()/(v.T.conj() @ x)) * v @ v.T.conj()
        a[k:, k:] -= H @ a[k:, k:]

        qk = np.identity(m).astype(q.dtype)
        H = (1. + (v.T.conj() @ x).conj()/(v.T.conj() @ x)) * v @ v.T.conj()
        # TODO: why does this not work?
        qk[k:, k:] -= H
        # = np.identity(v.shape[0]).astype(q.dtype) * v @ v.T.conj() USED TO BE ABOVE
        q = qk.T.conj() @ q

    if reduced:
        a = a[:mn]
        return q.T.conj()[:, :mn], a[:mn]#np.triu(a[:mn])
    else:
        return q.T.conj(), np.triu(a)


# TODO: separate them
def _householder(a):
    x = a[:, 0].reshape(-1, 1)
    e1 = np.zeros_like(x)
    e1[0] = 1.
    v = sign(x[0]) * norm(x) * e1 + x
    v /= norm(v)
    H = (1. + (v.T.conj() @ x).conj() / (v.T.conj() @ x)) * v @ v.T.conj()
    return H


def qr2(a, b=None, reduced=True, inplace=False):
    mn = min(a.shape)
    m, _ = a.shape
    q = np.identity(m).astype(a.dtype)

    if not inplace:
        a = a.copy()

    H = np.zeros(q.shape).astype(a.dtype)

    for k in range(mn):
        H[k:, k:] = _householder(a[k:, k:])
        # a[k:, k:] -= H[k:, k:] @ a[k:, k:]
        qk = np.identity(m).astype(a.dtype)
        qk[k:, k:] -= H[k:, k:]
        q = qk.T.conj() @ H
        a[k:, k:] -= H[k:, k:] @ a[k:, k:]

        try:
            b[k:] -= H @ b[k:]
        except TypeError:
            continue

    if reduced:
        q = q.T.conj()[:, :mn]
        a = a[:mn]

    if b is not None and not inplace:
        return q, a, b
    elif b is None and not inplace:
        return q, a
    else:
        return q


def solve(a, b):
    r, b = householder_ab(a, b)
    return solve_triu(r, b)


# TODO: add arg to return x as column or row?
def solve_triu(r, b):
    assert np.allclose(r, np.triu(r))
    x = np.zeros((r.shape[1], 1))
    x[-1] = b[-1] / r[-1, -1]
    for j in range(r.shape[0] - 2, -1, -1):
        x[j] = (b[j] - r[j, j + 1:] @ x[j + 1:]) / r[j, j]
    return x


def solve_tril(l, b):
    assert np.allclose(l, np.tril(l))
    x = np.zeros((l.shape[1], 1))
    x[0] = b[0] / l[0, 0]
    for j in range(1, l.shape[0], 1):
        x[j] = (b[j] - l[j, :j] @ x[:j]) / l[j, j]
    return x


def solve_diag(d, b):
    assert np.allclose(d, np.diag(np.diag(d)))
    # TODO: x = np.array([d[i, i] for i in range(len(d))])
    x = b / np.diag(d).reshape(b.shape)
    return x


def poly1d(c, x):
    # return np.array([1**i for i in range(len(c))]) @ c
    return np.column_stack([x**i for i in range(len(c))]) @ c


def polyfit(x, y, deg, plot=False):
    vander = np.column_stack([x**i for i in range(deg + 1)])
    # scale to improve condition number
    scale = np.sqrt((vander*vander).sum(axis=0))
    vander /= scale
    q, r = qr(vander)
    b = q.T.conj() @ y
    c = solve_triu(r, b)  # scaled
    residuals = norm(y - c.reshape(-1) @ vander.T.conj()) ** 2
    c = (c.T/scale).T  # original scale

    if plot:
        import matplotlib.pyplot as plt

        xx = np.linspace(np.min(x), np.max(x), 100)

        # TODO: don't need to sort usually
        x_sorted, y_sorted = zip(*sorted(zip(xx, poly1d(c, xx))))
        plt.plot(x, y, '.', label='Data')
        plt.plot(x_sorted, y_sorted, label='Best fit')
        plt.legend(loc='best')
        plt.title('Least squares polynomial regression of deg {}'.format(deg))
        plt.show()

    return c, residuals


class Polynomial():
    def __init__(self, c):
        pass
