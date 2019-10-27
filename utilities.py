import numpy as np


def norm(x):
    return np.sqrt(np.sum(np.square(x)))


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
    q = np.identity(m)

    if not inplace:
        a = a.copy()

    for k in range(mn):
        x = a[k:, k].reshape(-1, 1)
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = sign(x[0]) * norm(x) * e1 + x
        v /= norm(v)
        a[k:, k:] -= 2 * v @ v.T.conj() @ a[k:, k:]

        qk = np.identity(m)
        qk[k:, k:] = np.identity(v.shape[0]) - 2 * v @ v.T.conj()
        q = qk @ q

    if reduced:
        a = a[:mn]
        return q.T.conj()[:, :mn], a[:mn]#np.triu(a[:mn])
    else:
        return q.T.conj(), np.triu(a)


def solve(a, b):
    r, b = householder_ab(a, b)
    x = np.zeros((r.shape[1], 1))
    x[-1] = b[-1] / r[-1, -1]
    for j in range(r.shape[0] - 2, -1, -1):
        x[j] = (b[j] - r[j, j + 1:] @ x[j + 1:]) / r[j, j]
    return x


def lstsq(x, y, deg, resids = True, plot=False):
    vander = np.column_stack([x**i for i in range(deg + 1)])
    q, r = qr(vander)
    b = q.T.conj() @ y
    c = solve(r, b)
    print(c)
    p = np.poly1d(np.flip(c.flat))
    r = vander @ c - y
    print(np.absolute(r).sum())

    if resids:
        # r = np.sum([p(i) for i in c])
        pass

    if plot:
        import matplotlib.pyplot as plt

        xxplot = np.linspace(np.min(x), np.max(x), 100)
        # plt.plot(x, y, xxplot, vander @ xx)

        plt.plot(x, y, '.', xxplot, p(xxplot), '-')
        plt.show()
