import numpy as np
# import scipy.linalg
import utilities

# data1 = np.genfromtxt('./readings.csv', delimiter=',', dtype=float)
# # data2 = np.genfromtxt('./readings2.csv', delimiter=',', dtype=float)
# print(np.shape(data1))
# light = data1[:, 0]
# temp = data1[:, 1]

m, n = (5, 5)
np.random.seed(1)
An = np.random.rand(m, n).astype(np.float_) #+ 1j * np.random.rand(m, n)
qn, rn = np.linalg.qr(An)
qrn = qn @ rn


# print(np.array_equal(np.conj(np.transpose(qn)) @ qn, qn @ np.conj(np.transpose(qn))))
# print(np.array_equal(np.identity(qn.shape[0]), qn.T.conj() @ qn))
# print(np.absolute(A - qn @ rn).sum())
# print(qn.T.conj() @ qn)

np.random.seed(1)
A = np.random.rand(m, n).astype(np.float_) #+ 1j * np.random.rand(m, n)
q, r = utilities.householder_qr(A)
np.random.seed(1)
A = np.random.rand(m, n).astype(np.float_)
qr = q @ r

# print(A)
# print(qr)
# print(qrn)
