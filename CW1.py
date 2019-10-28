import numpy as np
import utilities

# # data2 = np.genfromtxt('./readings2.csv', delimiter=',', dtype=float)

m, n = (8, 5)
np.random.seed(1)
An = np.random.rand(m, n).astype(np.complex_) + 1j * np.random.rand(m, n)
qn, rn = np.linalg.qr(An)
# qrn = qn @ rn
print('qn: ')
print(np.round(qn, 3))
print('rn: ')
print(np.round(rn, 3))

# print(np.array_equal(np.conj(np.transpose(qn)) @ qn, qn @ np.conj(np.transpose(qn))))
# print(np.array_equal(np.identity(qn.shape[0]), qn.T.conj() @ qn))
# print(np.absolute(A - qn @ rn).sum())
# print(qn.T.conj() @ qn)

np.random.seed(1)
A = np.random.rand(m, n).astype(np.complex_) + 1j * np.random.rand(m, n)
q, r = utilities.qr(A, inplace=False)

# print(A)
print('q: ')
print(np.round(q, 3))
print('r: ')
print(np.round(r, 3))

# LEAST SQUARES

data1 = np.genfromtxt('./readings.csv', delimiter=',', dtype=float)
light = data1[:, 0]
temp = data1[:, 1]

# x = np.array([1, 2, 3, 4])
z, res, rank, sing_values, _ = np.polyfit(light, temp, 4, full=True)
# print('z: ', z)
# print('res from numpy: ', res)
# print('rank: ', rank)

# xxplot = np.linspace(np.min(light), np.max(light), 100)
p = np.poly1d(z)
y = p(light)
# print('y from numpy: ', y)
# plt.plot(light, temp, '.', xxplot, y)
# plt.show()
utilities.polyfit(light, temp, 4, plot=True)
