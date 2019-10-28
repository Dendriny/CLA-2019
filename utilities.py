import numpy as np


def norm(x):
    # return np.sqrt(np.sum(np.square(np.real(x)) + np.square(np.imag(x))))
    return np.sqrt(x.T.conj() @ x)

def sign(x):
    return 1 if x >= 0 else -1


def householder_complex(a: np.ndarray) -> np.ndarray:
    # for k in range(n - (m == n)):
    m, n = a.shape
    for k in range(min(m, n)):
        x = a[k:, k].reshape(-1, 1)
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = sign(x[0]) * norm(x) * e1 + x
        v /= norm(v)
        a[k:, k:] -= 2 * v @ (np.conj(np.transpose(v)) @ a[k:, k:])
    return np.triu(a)


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


# newest
def qr(a, reduced=True, inplace=True):
    mn = min(a.shape)
    m, _ = a.shape
    q = np.identity(m).astype(a.dtype)
    a_orig = a
    if not inplace:
        a = a.copy()

    for k in range(mn):
        x = a[k:, k].reshape(-1, 1)
        e1 = np.zeros_like(x)
        e1[0] = 1.
        v = sign(x[0]) * norm(x) * e1 + x
        # v = x[0]/np.abs(x[0]) * norm(x) * e1 + x
        v /= norm(v)
        # a[k:, k:] -= 2 * v @ v.T.conj() @ a[k:, k:] USED TO BE BELOW
        a[k:, k:] -= (1. + (v.T.conj() @ x).conj()/(v.T.conj() @ x)) * v @ v.T.conj() @ a[k:, k:]

        qk = np.identity(m).astype(q.dtype)
        qk[k:, k:] -= (1. + (v.T.conj() @ x).conj()/(v.T.conj() @ x)) * v @ v.T.conj()
        # = np.identity(v.shape[0]).astype(q.dtype) * v @ v.T.conj() USED TO BE ABOVE
        q = qk.T.conj() @ q

    if reduced:
        a = a[:mn]
        return q.T.conj()[:, :mn], a[:mn]#np.triu(a[:mn])
    else:
        return q.T.conj(), np.triu(a)


def solve(a, b, desc=None):

    if desc is None:
        if np.allclose(a, np.triu(a)):
            desc = 'upper'
        elif np.allclose(a, np.tril(a)):
            desc = 'lower'
        elif np.allclose(a, np.diag(np.diag(a))):
            desc = 'diagonal'
        else:
            r, b = householder_ab(a, b)

    r, b = householder_ab(a, b)
    x = np.zeros((r.shape[1], 1))
    x[-1] = b[-1] / r[-1, -1]
    for j in range(r.shape[0] - 2, -1, -1):
        x[j] = (b[j] - r[j, j + 1:] @ x[j + 1:]) / r[j, j]
    return x


def _poly1d(c, x):
    # return np.array([1**i for i in range(len(c))]) @ c
    return np.column_stack([x**i for i in range(len(c))]) @ c


def polyfit(x, y, deg):
    pass


def lstsq(x, y, deg, resids = True, plot=False):
    vander = np.column_stack([x**i for i in range(deg + 1)])
    print(vander.shape, y.shape, x.shape)
    # scale lhs to improve condition number
    m, rr, rank, s = np.linalg.lstsq(vander, y, rcond=None)
    residuals = norm(y - m @ vander.T.conj())**2
    print('residuals from np: ', residuals)
    scale = np.sqrt((vander*vander).sum(axis=0))
    vander /= scale
    q, r = qr(vander, inplace=False)
    b = q.T.conj() @ y
    c = solve(r, b)  # scaled
    residuals = norm(y - c.reshape(-1) @ vander.T.conj()) ** 2
    print('residuals from me scaled: ', residuals)
    c = (c.T/scale).T  # original scale
    print('c unscaled: ', c)
    p = np.poly1d(np.flip(c.flat))
    modely = p(x)
    myp = _poly1d(c, x).reshape(-1)
    modely -= y
    # print('mymodely: ', poly1d(c, x))
    print('a: ', np.absolute(modely).sum())
    # print(r)
    print('b: ', np.sqrt((modely*modely).sum()))
    print(myp.shape, y.shape)
    resid = myp - y
    print(np.sqrt((resid*resid).sum()))

    if resids:
        # r = np.sum([p(i) for i in c])
        pass

    if plot:
        import matplotlib.pyplot as plt

        xxplot = np.linspace(np.min(x), np.max(x), 100)
        # plt.plot(x, y, xxplot, vander @ xx)

        # plt.plot(x, y, '.', xxplot, poly1d(c, xxplot), '-', xxplot, p(xxplot), '.')
        plt.plot(x, y, '.', x, _poly1d(c, x), '-')
        plt.show()


class Polynomial():
    def __init__(self, c):
        pass

