import numpy as np
import utilities
import matplotlib.pyplot as plt
from unittest import TestLoader, TextTestRunner
import tests.test_qr

# UNIT TESTS
print('Unit testing:')
suite = TestLoader().loadTestsFromModule(tests.test_qr)
TextTestRunner().run(suite)

print('********************************')

# PRECISION TESTING
print('Problem 1: QR Factorisation')
print('Validation #8:')
A = np.array([[3., 3., 2.],
              [4., 4., 1.],
              [0., 6., 2.],
              [0., 8., 1.]])

Q = np.array([[3/5, 0., 2**1.5 / 5],
              [4/5, 0., -3 * 2**0.5 / 10],
              [0., 3/5, 2**1.5 / 5],
              [0., 4/5, -3 * 2**0.5 / 10]])
R = np.array([[5., 5., 2.],
              [0., 10., 2.],
              [0., 0., 2**0.5]])

q, r = utilities.qr(A)
Qdiff = np.sum(np.absolute(Q) - np.absolute(q))
Rdiff = np.sum(np.absolute(R) - np.absolute(r))
AQRdiff = np.sum(np.absolute(A) - np.absolute(q @ r))
QRdiff = np.sum(np.absolute(Q @ R) - np.absolute(q @ r))
print('Q - q: ', Qdiff)
print('R - r: ', Rdiff)
print('A - qr: ', AQRdiff)
print('QR - qr: ', QRdiff)

print('********************************')

# PROBLEM 2
print('Problem 2: Least squares polynomial fit')
data1 = np.genfromtxt('./readings.csv', delimiter=',', dtype=float)
print('Loaded data.')
light = data1[:, 0]
temp = data1[:, 1]
residuals = []
coeffs = []

light, temp = zip(*sorted(zip(light, temp)))
light = np.asarray(light)
temp = np.asarray(temp)

for deg in range(1, 10):
    print('Computing least squares polynomial for deg ', deg)
    c, res = utilities.polyfit(light, temp, deg)
    residuals.append(res)
    coeffs.append(c)
    print('coefficients: ', c.reshape(-1))
    print('sum of squared residuals: ', res)
    yy = utilities.poly1d(c, light).reshape(-1)
    R = 1 - np.sum([(temp[i] - yy[i]) ** 2 for i in range(len(yy))])
    print('R: ', np.round(R, 7))
    diff = np.abs(yy) - np.abs(temp)
    print('Plotting residuals for deg {}:'.format(deg))
    plt.plot(light, diff, '.', label='deg {}'.format(deg))
    plt.legend(loc='best')
    plt.xlabel('Light intensity')
    plt.ylabel(r'$y - \hat{y} (^{\circ}C)$')
    plt.show()

print('Plotting total residuals:')
plt.plot([i for i in range(1, 10)], residuals)
plt.ylabel('Residual')
plt.xlabel('Polynomial degree')
plt.show()

print('Plotting polynomials:')
xx = np.linspace(np.min(light), np.max(light), 100)
plt.plot(light, temp, '.', label='Observations')
for i in range(2, 5):
    plt.plot(xx, utilities.poly1d(coeffs[i], xx), label='deg {}'.format(i+1))
plt.xlabel('Light intensity')
plt.ylabel(r'$\Delta T (^{\circ}C)$')
plt.legend(loc='best')
plt.show()

print('********************************')

# PROBLEM 3
print('Problem 3: analysing a dataset')
data2 = np.genfromtxt('./readings2.csv', delimiter=',', dtype=float)
print('Loaded data.')
q, r = utilities.qr(data2)

print('R:')
print(np.round(r, 3))

a = q[:, :2] @ r[:2, :]
print('Quick validation of results: A = Q1R1 ?', np.allclose(a, data2))
