import sys, unittest
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmmstrata.model import GPT
from hmmstrata.ablate import probs_with_ablation, orth_basis


class TestAblate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = GPT(6, ctx=16).eval()
        self.x = np.random.default_rng(0).integers(0, 6, size=(64, 16)).astype(np.int64)

    def test_no_ablation_equals_forward(self):
        with torch.no_grad():
            ref = torch.softmax(self.m(torch.from_numpy(self.x)), -1).numpy()
        got = probs_with_ablation(self.m, self.x, 2, None, np.zeros(64), "cpu")
        self.assertTrue(np.allclose(ref, got, atol=1e-6))

    def test_ablating_a_direction_removes_its_variance(self):
        Q = orth_basis(np.random.default_rng(1).normal(size=(64, 3)))
        self.assertTrue(np.allclose(Q.T @ Q, np.eye(3), atol=1e-8))
        A = np.random.default_rng(2).normal(size=(500, 64)); mu = A.mean(0)
        out = A - (A - mu) @ Q @ Q.T
        self.assertLess(np.abs((out - out.mean(0)) @ Q).max(), 1e-9)

    def test_orth_basis_is_orthogonal_to_excluded_span(self):
        rng = np.random.default_rng(3)
        Qb = orth_basis(rng.normal(size=(64, 3)))
        Qh = orth_basis(rng.normal(size=(64, 8)), against=Qb)
        self.assertLess(np.abs(Qb.T @ Qh).max(), 1e-9)


if __name__ == "__main__":
    unittest.main()
