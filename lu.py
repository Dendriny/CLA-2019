from scipy.sparse import csr_matrix as csr
import numpy as np
from rexi_coefficients import RexiCoefficients
from scipy.sparse.linalg import eigs, spsolve


def get_banded(a: np.ndarray, p, q):
    """
    Converts the matrix a into a banded scipy.sparse.csr_matrix object with
    lower and upper bandwidths p and q, respectively. Returns a CSR matrix
    of the same shape and banded entries as a.
    """

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i > j + p or j > i + q:
                a[i, j] = 0.

    return csr(a)


def bandedLU(M: csr, ml, mu):
    """
    Computes standard LU decomposition of a class 'scipy.sparse.csr.csr_matrix'
    banded square matrix M with lower and upper bandwidths ml and mu, respectively.
    Returns L and U as sparse CSR matrices.
    """

    m = M.shape[0]
    u = M.copy()  # can remove to act directly on M

    # Allocating memory to store nnzl number of non-zero entries of L
    nnzl = int(m*(ml + 1) - ml*(ml + 1)/2)
    l_row = np.zeros(nnzl).astype(np.int_)
    l_val = np.ones(nnzl).astype(M.dtype)

    for i in range(m):
        l_row[i] = i
    l_col = l_row.copy()
    count = i + 1  # counter for the next entry of L

    for k in range(m - 1):
        column_entries_ind = u.indptr[k] + (u.indices[u.indptr[k]:u.indptr[min(k + ml + 1, m)]] == k).nonzero()[0]
        for i, ind in enumerate(column_entries_ind[1:]):
            l = u.data[ind] / u.data[column_entries_ind[0]]
            l_val[count] = l; l_col[count] = k; l_row[count] = int(k + i + 1)
            count += 1

            b = min(mu + 1, m - k)
            u.data[ind + 1:ind + b] -= l * u.data[column_entries_ind[0] + 1:column_entries_ind[0] + b]
            u.data[ind] = 0.

    u.eliminate_zeros()
    l = csr((l_val, (l_row, l_col)))

    return l, u


def banded_solve_backward(u: csr, b, mu):
    """
    Performs back substitution to solve lx=b, where u is an upper triangular banded
    matrix with upper bandwidth mu in CSR format as scipy.sparse.csr_matrix object.
    Returns the solution array x.
    """

    m = len(b)
    x = np.zeros(m).astype(u.dtype)
    x[-1] = b[-1] / u.data[-1]
    for j in range(m - 2, -1, -1):
        x[j] = (b[j] - u.data[u.indptr[j] + 1:u.indptr[j + 1]] @ x[j+1:min(m, j + mu + 1)]) \
               / u.data[u.indptr[j]]
    return x


def banded_solve_forward(l: csr, b, ml):
    """
    Performs forward substitution to solve lx=b, where l is a lower triangular banded
    matrix with lower bandwidth ml in CSR format as scipy.sparse.csr_matrix object.
    Returns the solution array x.
    """

    m = len(b)
    x = np.zeros(m).astype(l.dtype)
    x[0] = b[0] / l.data[0]
    for j in range(1, len(b)):
        x[j] = (b[j] - l.data[l.indptr[j]:l.indptr[j + 1] - 1] @ x[max(0, j - ml):min(j, m)]) \
               / l.data[l.indptr[j + 1] - 1]
    return x


def banded_solve(a: csr, b, ml, mu):
    """
    Solves ax=b for x, where a is a square banded matrix with lower and upper bandwidth
    ml and mu, respectively, stored in CSR format as scipy.sparse.csr_matrix object.
    Returns the solution array x.
    """

    l, u = bandedLU(a, ml, mu)
    y = banded_solve_forward(l, b, ml)

    return banded_solve_backward(u, y, mu)
    

def solve_wave(h, dx, t):
    """
    Solves the wave equation u_tt - u_xx = 0 using exponential integrators.
    The function is evaluated at time t on the domain [0, h] discretized with
    a dx step in space. The boundary conditions are defined to u(0) = u(h) = 0.
    Initial conditions are u(0) = np.exp(-(x - 5)**2 / 0.2) - np.exp(-125)
    and du/dt = 0. Returns the solution array u(t).
    """

    n = int(h / dx - 1)
    v = np.linspace(dx, h, n)
    I, K, L = _get_kl(n, dx)
    U0 = np.exp(-(v - 5) ** 2 / 0.2) - np.exp(-125)

    mu_max = eigs(L, k=1, which='LM', return_eigenvectors=False)  # largest magnitude
    hM = 1.1 * t * abs(mu_max)  # hM should be slightly bigger than this
    M = 500
    h = hM/M
    a, b = RexiCoefficients(h, M)

    U_T = np.zeros(len(v)).astype(np.complex_)
    for j in range(len(b)):
        u = a[j] * I + t ** 2 / a[j] * K  # inefficient a[j] * I
        U_j = banded_solve(u, U0, 1, 1)
        U_T += b[j] * U_j

    return U_T


def rk2(h, dx, t, dt):
    """
    Solves the wave equation u_tt - u_xx = 0 using 2nd order Runge-Kutta
    time stepping. The function is evaluated at time t on the domain [0, h]
    discretized with a dx step in space and step dt in time. The boundary
    conditions are defined to u(0) = u(h) = 0. Initial conditions are
    u(0) = np.exp(-(x - 5)**2 / 0.2) - np.exp(-125) and du/dt = 0.
    Returns the list of evaluated times t' and list of solution arrays u(t').
    """

    n = int(h / dx - 1)
    v = np.array([dx * i for i in range(1, n + 1)])
    V = np.exp(-(v - 5) ** 2 / 0.2) - np.exp(-125)
    w = np.zeros(len(v))
    U = np.array([V, w]).reshape(-1, 1)  # Initial U(0)
    _, _, L = _get_kl(n, dx)

    t_current = 0.
    tl = [t_current]
    Ul = [U]

    while t_current < t:
        U_half = U + dt / 2 * _f(L, U)
        U = U + dt * _f(L, U_half)

        t_current += dt

        tl.append(t_current)
        Ul.append(U)

    return tl, Ul


def _f(l, u):
    return l.dot(u)


def _get_kl(n, dx):
    rows2 = [i for i in range(n)]
    lower1rows = [i + 1 for i in range(n - 1)]
    rows = rows2 + rows2[:-1] + lower1rows
    cols = rows2 + lower1rows + rows2[:-1]
    vals = np.array([2.] * n + [-1.] * (n - 1) * 2) / dx ** 2
    k = csr((vals, (rows, cols)))

    vals = (np.array([2.] * n + [-1.] * (n - 1) * 2) * -1 / dx ** 2).tolist()
    iden = [1.] * n
    rowsid = (np.array(rows2) + n).tolist()
    i = csr((iden, (rows2, rows2)))

    l = csr((vals + iden, ((np.array(rows) + n).tolist() + rows2, cols + rowsid)))

    return i, k, l
