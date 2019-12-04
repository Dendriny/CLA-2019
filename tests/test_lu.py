from unittest import TestCase
from lu import bandedLU, get_banded, banded_solve
import numpy as np
import scipy.sparse as sp


class Test(TestCase):

    def test_lu(self):
        bandwidths = [(2, 4), (4, 2), (3, 3)]
        for p, q in bandwidths:
            m = 20
            np.random.seed(2)
            a = np.random.rand(m, m).astype(np.complex_) + 1j * np.random.rand(m, m)
            A = get_banded(a, p, q)

            L, U = bandedLU(A, p, q)

            # Tests are number-referenced to the test descriptions in section 2
            self.assertAlmostEqual(0, np.absolute(A - L.dot(U)).sum())  # 1
            self.assertTrue(np.array_equal(L.todense(), sp.tril(L).todense()))  # 2
            self.assertTrue(np.array_equal(U.todense(), sp.triu(U).todense()))  # 3
            diag_product_L = np.prod(L.diagonal())
            self.assertAlmostEqual(diag_product_L, np.linalg.det(L.todense()))  # 4
            diag_product_U = np.prod(U.diagonal())
            self.assertAlmostEqual(diag_product_U, np.linalg.det(U.todense()))  # 4

            b = np.random.rand(m, 1).astype(np.float_)
            self.assertTrue(np.allclose(sp.linalg.spsolve(A, b), banded_solve(A, b, p, q).flatten(), 1e-15))  # 5
