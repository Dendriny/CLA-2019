from unittest import TestCase
from utilities import norm, qr
import numpy as np


class TestNorm(TestCase):
    def test_norm(self):
        v = np.array([[5.], [4.], [1.]])
        actual = norm(v)
        expected = np.sqrt(5.**2 + 4.**2 + 1.)
        self.assertEqual(expected, actual)
        # TODO: add complex case

    def test_qr_real(self):
        m, n = (8, 8)
        np.random.seed(1)
        a = np.random.rand(m, n).astype(np.float_)
        q, r = qr(a, inplace=False)
        # np.random.seed(1)
        # a = np.random.rand(m, n).astype(np.float_)
        self.assertTrue(np.allclose(a, q @ r))
        self.assertAlmostEqual(np.abs(1), np.linalg.det(q))
        self.assertTrue(np.allclose(q.T @ q, q @ q.T))
        self.assertTrue(np.allclose(np.identity(q.shape[0]), q @ q.T))

    def test_qr_complex(self):
        m, n = (8, 8)
        np.random.seed(1)
        a = np.random.rand(m, n).astype(np.complex_) + 1j * np.random.rand(m, n)
        q, r = qr(a)
        np.random.seed(1)
        a = np.random.rand(m, n).astype(np.complex_) + 1j * np.random.rand(m, n)
        self.assertAlmostEqual(0, np.absolute(a - q @ r).sum())
        # self.assertAlmostEqual(np.abs(1), np.linalg.det(q))
        self.assertAlmostEqual(0, np.absolute(q.T.conj() @ q - q @ q.T.conj()).sum())
        self.assertAlmostEqual(0, np.absolute(np.identity(q.shape[0]) - q @ q.T.conj()).sum())
