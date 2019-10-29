from unittest import TestCase
from utilities import norm, qr, mgs, solve
import numpy as np


class Test(TestCase):
    def test_norm(self):
        v_r = np.array([[5.], [4.], [1.]])
        self.assertEqual(np.sqrt(5.**2 + 4.**2 + 1.), norm(v_r))
        v_c = v_r + 1j * np.array([[1.], [0.], [3.]])
        self.assertEqual(np.sqrt(5.**2 + 1. + 4.**2 + 1. + 3**2), norm(v_c))

    def test_qr_real(self):
        shapes = [(8, 5), (5, 8), (8, 8)]
        for m, n in shapes:
            np.random.seed(1)
            a = np.random.rand(m, n).astype(np.float_)
            q, r = qr(a, reduced=False, inplace=False)
            self.assertAlmostEqual(0, np.absolute(a - q @ r).sum())
            if m == n:
                self.assertAlmostEqual(np.abs(1), np.linalg.det(q))
            self.assertTrue(np.allclose(q.T @ q, q @ q.T))
            self.assertTrue(np.allclose(np.identity(q.shape[0]), q @ q.T))
            mgs_q, mgs_r = mgs(a)
            self.assertTrue(np.allclose(mgs_q @ mgs_r, q @ r))

        a = np.array([[1., 1., 1.],
                      [0., 2., 5.],
                      [2., 5., -1.]])
        b = np.array([6., -4., 27.]).reshape(-1, 1)
        x_expected = np.array([5, 3, -2]).reshape(-1, 1)
        x_actual = solve(a, b)
        self.assertTrue(np.allclose(x_expected, x_actual))

    def test_qr_complex(self):
        shapes = [(8, 5), (5, 8), (8, 8)]
        for m, n in shapes:
            np.random.seed(1)
            a = np.random.rand(m, n).astype(np.complex_) + 1j * np.random.rand(m, n)
            q, r = qr(a, reduced=False, inplace=False)
            self.assertAlmostEqual(0, np.absolute(a - q @ r).sum())
            self.assertAlmostEqual(np.abs(1), norm(np.linalg.det(q)))
            if m == n:
                self.assertAlmostEqual(np.linalg.det(a.T.conj()), np.linalg.det(a).conj())
            self.assertTrue(np.allclose(q.T.conj() @ q, q @ q.T.conj()))
            self.assertTrue(np.allclose(np.identity(q.shape[0]), q @ q.T.conj()))
            mgs_q, mgs_r = mgs(a)
            self.assertTrue(np.allclose(mgs_q @ mgs_r, q @ r))
        a = np.array([[5. + 5.j, -5.j, 2.],
                      [-4.j, 8. + 8.j, -6.],
                      [0., -6., 10.]])
        b = np.array([3., 5. + 4.j, -2.]).reshape(-1, 1)
        x_expected = np.array([[0.6149065 + 0.1621229j],
                               [0.7862721 + 0.0385111j],
                               [0.2717633 + 0.0231067j]])
        x_actual = solve(a, b)
        self.assertAlmostEqual(x_expected.all(), x_actual.all())
