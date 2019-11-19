from unittest import TestCase
from utilities import norm, qr, mgs, solve
import numpy as np


class Test(TestCase):
    def test_norm(self):
        v_r = np.array([[5.], [4.], [1.]])
        self.assertEqual(np.sqrt(5.**2 + 4.**2 + 1.), norm(v_r))
        v_c = v_r + 1j * np.array([[1.], [0.], [3.]])
        self.assertEqual(np.sqrt(5.**2 + 1. + 4.**2 + 1. + 3**2), norm(v_c))

# tests are referenced to the validation points in section 1
    def test_qr_real(self):
        shapes = [(20, 15), (15, 20), (20, 20)]
        for m, n in shapes:
            np.random.seed(1)  # always get the same random
            a = np.random.rand(m, n).astype(np.float_)
            q, r = qr(a, reduced=False)
            self.assertAlmostEqual(0, np.absolute(a - q @ r).sum())  # val.1
            if m == n:
                self.assertAlmostEqual(q.T.all(), np.linalg.inv(q).all())  # val.6
            self.assertAlmostEqual(1, np.abs(np.linalg.det(q)))  # val.3
            self.assertTrue(np.allclose(q.T @ q, q @ q.T))  # val.4
            self.assertTrue(np.allclose(np.identity(q.shape[0]), q @ q.T))  # val.5
            self.assertTrue(np.allclose(np.triu(r), r))  # val.7
            q, r = qr(a)
            mgs_q, mgs_r = mgs(a)
            self.assertTrue(np.allclose(np.absolute(mgs_q), np.absolute(q)))  # val.2
            self.assertTrue(np.allclose(np.absolute(mgs_r), np.absolute(r)))  # val.2

        a = np.array([[1., 1., 1.],
                      [0., 2., 5.],
                      [2., 5., -1.]])
        b = np.array([6., -4., 27.]).reshape(-1, 1)
        x_expected = np.array([5, 3, -2]).reshape(-1, 1)
        x_actual = solve(a, b)
        self.assertTrue(np.allclose(x_expected, x_actual))  # val.8

    def test_qr_complex(self):
        shapes = [(20, 15), (15, 20), (20, 20)]
        for m, n in shapes:
            np.random.seed(1)  # always get the same random
            a = np.random.rand(m, n).astype(np.complex_) + 1j * np.random.rand(m, n)
            q, r = qr(a, reduced=False)
            self.assertAlmostEqual(0, np.absolute(a - q @ r).sum())  # val.1
            if m == n:
                self.assertAlmostEqual(q.T.conj().all(), np.linalg.inv(q).all())  # val.6
            self.assertAlmostEqual(1, norm(np.linalg.det(q)))  # val. 3
            self.assertTrue(np.allclose(q.T.conj() @ q, q @ q.T.conj()))  # val.4
            self.assertTrue(np.allclose(np.identity(q.shape[0]), q @ q.T.conj()))  # val.5
            self.assertTrue(np.allclose(np.triu(r), r))  # val.7
            q, r = qr(a)
            mgs_q, mgs_r = mgs(a)
            self.assertTrue(np.allclose(np.absolute(mgs_q), np.absolute(q)))  # val.2
            self.assertTrue(np.allclose(np.absolute(mgs_r), np.absolute(r)))  # val.2

        a = np.array([[5. + 5.j, -5.j, 2.],
                      [-4.j, 8. + 8.j, -6.],
                      [0., -6., 10.]])
        b = np.array([3., 5. + 4.j, -2.]).reshape(-1, 1)
        x_expected = np.array([[0.6149065 + 0.1621229j],
                               [0.7862721 + 0.0385111j],
                               [0.2717633 + 0.0231067j]])
        x_actual = solve(a, b)
        self.assertAlmostEqual(x_expected.all(), x_actual.all())  # val.8
