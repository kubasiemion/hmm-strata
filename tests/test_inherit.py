import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmmstrata.process import make
from hmmstrata import exact
from hmmstrata.inherit import inheritance_from_estimates, jacobian_push


class TestInherit(unittest.TestCase):
    def setUp(self):
        self.h = make("sync_strata", dest_emission=True)
        self.x = exact.sample(self.h, 40000, 16, np.random.default_rng(0))[0]
        self.eta, _, _ = exact.beliefs(self.h, self.x)

    def test_jacobian_matches_finite_difference(self):
        h, rng = self.h, np.random.default_rng(1)
        eta = rng.dirichlet(np.ones(4), 5); x = np.array([0, 1, 2, 3, 0]); e = rng.normal(size=(5, 4)) * 1e-6
        def f(b):
            u = np.einsum("ni,nij->nj", b, h.T[x]); return u / u.sum(1, keepdims=True)
        fd = f(eta + e) - f(eta)
        self.assertTrue(np.allclose(jacobian_push(h, eta, x, e), fd, atol=1e-10))

    def test_recursive_model_inherits_parallel_does_not(self):
        h, x, eta = self.h, self.x, self.eta
        rng = np.random.default_rng(2)
        # parallel: truth + independent noise at every position
        par = eta + rng.normal(size=eta.shape) * 0.05
        # recursive: run the TRUE update on its own estimate; at an s the estimate is re-seeded with
        # an error (true point + noise), so the error is then carried through the update
        rec = np.empty_like(eta); cur = rng.dirichlet(np.ones(4), len(x))
        for t in range(x.shape[1]):
            u = np.einsum("ni,nij->nj", cur, h.T[x[:, t]]); cur = u / u.sum(1, keepdims=True)
            hit = x[:, t] == 4
            pert = np.abs(cur + rng.normal(size=cur.shape) * 0.05); pert /= pert.sum(1, keepdims=True)
            cur = np.where(hit[:, None], pert, cur); rec[:, t] = cur
        p = inheritance_from_estimates(h, x, eta, par)
        r = inheritance_from_estimates(h, x, eta, rec)
        # depth 1: recursive carries its error through the (nonlinear) true update; parallel does not
        self.assertGreater(r[1][0], 0.5)
        self.assertLess(abs(p[1][0]), 0.05)


if __name__ == "__main__":
    unittest.main()
