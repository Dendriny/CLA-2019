import numpy as np


def norm(x):
    # return np.sqrt(np.sum(np.square(np.real(x)) + np.square(np.imag(x))))
    return np.sqrt(x.T.conj() @ x)


def sign(x):
    return 1 if np.real(x) >= 0 else -1


def qr_old(a: np.ndarray, reduced=True, inplace=True):
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
        h = (1. + (v.T.conj() @ x).conj()/(v.T.conj() @ x)) * v @ v.T.conj()
        a[k:, k:] -= h @ a[k:, k:]

        qk = np.identity(m).astype(q.dtype)
        # h = (1. + (v.T.conj() @ x).conj()/(v.T.conj() @ x)) * v @ v.T.conj()
        # TODO: why does this not work? Have to add .T.conj()
        qk[k:, k:] -= h.T.conj()
        # = np.identity(v.shape[0]).astype(q.dtype) * v @ v.T.conj() USED TO BE ABOVE
        q = qk.T.conj() @ q

    if reduced:
        # TODO: pythonic
        return q.T.conj()[:, :mn], a[:mn]
    else:
        return q.T.conj(), np.triu(a)


def _householder(a):
    x = a[:, 0].reshape(-1, 1)
    e1 = np.zeros_like(x)
    e1[0] = 1.
    v = sign(x[0]) * norm(x) * e1 + x
    v /= norm(v)
    h = (1. + (v.T.conj() @ x).conj() / (v.T.conj() @ x)) * v @ v.T.conj()
    return h


def qr(a, b=None, reduced=True, inplace=False):
    mn = min(a.shape)
    m, _ = a.shape
    q = np.identity(m).astype(a.dtype)

    if not inplace:
        a = a.copy()

    h = np.zeros(q.shape).astype(a.dtype)

    for k in range(mn):
        h[k:, k:] = _householder(a[k:, k:])
        a[k:, k:] -= h[k:, k:] @ a[k:, k:]
        qk = np.identity(m).astype(a.dtype)
        qk[k:, k:] -= h[k:, k:].T.conj()
        q = qk.T.conj() @ q

        try:
            b[k:] -= h[k:, k:] @ b[k:]
        except TypeError:
            continue

    if reduced:
        q = q.T.conj()[:, :mn]
        a = a[:mn]

    if b is not None and not inplace:
        return q, a, b
    elif b is None and not inplace:
        return q, a
    elif b is not None:
        return q, b
    else:
        return q


def solve(a, b):
    _, r, b = qr(a, b)
    return solve_triu(r, b)


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
    c = (c.T/scale).T  # rescale
    # c = c.reshape(-1)  # row vector for user convenience

    if plot:
        import matplotlib.pyplot as plt

        xx = np.linspace(np.min(x), np.max(x), 100)
        name = str(np.round(c[0, 0], 3))
        name += ''.join(['+' + str(np.round(j[0], 3)) + 'x^{}'.format(i+1) for i, j in enumerate(c[1:])])

        x_sorted, y_sorted = zip(*sorted(zip(xx, poly1d(c, xx))))
        plt.plot(x, y, '.', label='Data')
        plt.plot(x_sorted, y_sorted, label='{}'.format(name))
        plt.legend(loc='best')
        plt.title('Least squares polynomial regression of deg {}'.format(deg))
        plt.show()

    return c, residuals
