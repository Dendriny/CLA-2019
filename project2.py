import numpy as np
from lu import solve_wave, rk2, get_banded, bandedLU
from scipy.sparse import csr_matrix as csr
import matplotlib.pyplot as plt
import time
from unittest import TestLoader, TextTestRunner
import tests.test_lu


# Q1 Stability and conditioning
# Householder QR

print('******** Householder ********')

m = 20

np.random.seed(2)
B = np.random.randn(20, 20).astype(np.complex_) + 1j * np.random.randn(20, 20).astype(np.float_)
Q, _ = np.linalg.qr(B)

C = np.random.randn(20, 20).astype(np.complex_) + 1j * np.random.randn(20, 20).astype(np.float_)
R = np.triu(C)

A = Q @ R
Q2, R2 = np.linalg.qr(A)

a = np.linalg.norm(Q2 - Q)
b = np.linalg.norm(R2 - R)
c = np.linalg.norm(Q2 @ R2 - A)

print('Q2 - Q = ', a)
print('R2 - R = ', b)
print('Q2R2 - A = ', c)

an = a / np.linalg.norm(Q2)
bn = b / np.linalg.norm(R2)
cn = c / np.linalg.norm(Q2 @ R2)

print('Normalized Q2 - Q = ', an)
print('Normalized R2 - R = ', bn)
print('Normalized Q2R2 - A = ', cn)

# SVD

print('******** SVD ********')

U, _ = np.linalg.qr(np.random.randn(m, m))
V, _ = np.linalg.qr(np.random.randn(m, m))
S = np.abs(np.diag(np.diag(np.random.randn(m, m))))

A = U @ S @ V
U2, S2, V2 = np.linalg.svd(A)
S2 = np.diag(S2)
S2 = np.diag(np.diag(S2))

a = np.linalg.norm(U2 - U)
b = np.linalg.norm(S2 - S)
c = np.linalg.norm(V2 - V)
d = np.linalg.norm(U2 @ S2 @ V2 - A)

print('U2 - U = ', a)
print('S2 - S = ', b)
print('V2 - V = ', c)
print('U2S2V2 - A = ', d)

a /= np.linalg.norm(U2)
b /= np.linalg.norm(S2)
c /= np.linalg.norm(V2)
d /= np.linalg.norm(U2 @ S2 @ V2)

print('Normalized U2 - U = ', a)
print('Normalized S2 - S = ', b)
print('Normalized V2 - V = ', c)
print('Normalized U2S2V2 - A = ', d)


# Q2 LU factorisation of a sparse CSR matrix

print('******** LU factorisation ********')

p = 2  # i > j + ml
q = 4  # j > i + mu
np.random.seed(2)
A = np.random.rand(m, m).astype(np.complex_) + 1j * np.random.rand(m, m).astype(np.float_)
A = get_banded(A, p, q)
A = csr(A)

L, U = bandedLU(A, p, q)

# UNIT TESTS
print('Unit testing:')
suite = TestLoader().loadTestsFromModule(tests.test_lu)
TextTestRunner().run(suite)


# Q3 Exponential integrators

print('******** Exponential integrators ********')

tic = time.time()
U_T = solve_wave(10, 0.01, 2.5)
print('Time taken for REXI: ', (time.time() - tic) / 60, ' min')
plt.plot(np.arange(len(U_T)), U_T, '--', label='REXI')

tic = time.time()
tl, Ul = rk2(10, 0.01, 2.5, 0.0001)
print('Time taken for RK: ', time.time() - tic, ' s')
Ul_final = Ul[-1]
plt.plot(np.arange(len(Ul_final) / 2), Ul_final[:int(len(Ul_final) / 2)], ':', label='Runge-Kutta')

plt.legend(loc='best')
plt.xlabel('x')
plt.show()

Ul_final = Ul_final[:len(U_T)].flatten()

diff = abs(Ul_final) - abs(U_T)
print('RK - REXI: ', diff.sum())
plt.plot(np.arange(len(diff)), diff)
plt.show()
