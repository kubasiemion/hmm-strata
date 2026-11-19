import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hmmstrata.process import make, names
from hmmstrata import exact


class TestProcesses(unittest.TestCase):
    def test_all_registered_are_valid(self):
        for n in names():
            h = make(n)
            self.assertTrue(np.allclose(h.T.sum(axis=(0, 2)), 1.0))

    def test_sync_strata_ranks(self):
        h = make("sync_strata")
        r = h.token_ranks()
        self.assertEqual(list(r), [4, 4, 4, 4, 1, 2])      # folds full rank, s rank 1, f rank 2
        self.assertEqual(list(h.stratum_dims()[4:]), [0, 1])
        self.assertTrue(np.allclose(h.pi0 @ h.T.sum(axis=0), h.pi0))   # stationary

    def test_beliefs_normalised_and_probs_sum_to_one(self):
        h = make("sync_strata")
        rng = np.random.default_rng(0)
        toks, _ = exact.sample(h, 500, 17, rng)
        eta, probs, prior = exact.beliefs(h, toks)
        self.assertTrue(np.allclose(eta.sum(-1), 1) and np.allclose(probs.sum(-1), 1))
        self.assertAlmostEqual(prior.sum(), 1.0)

    def test_empirical_frequencies_match_predictions(self):
        h = make("sync_strata")
        rng = np.random.default_rng(1)
        toks, _ = exact.sample(h, 200000, 6, rng)
        eta, probs, prior = exact.beliefs(h, toks)
        freq0 = np.bincount(toks[:, 0], minlength=h.V) / len(toks)
        self.assertTrue(np.allclose(freq0, prior, atol=0.005))
        # P(token 4 at position 3) averaged over predictions at position 2
        freq3 = np.bincount(toks[:, 3], minlength=h.V) / len(toks)
        self.assertTrue(np.allclose(freq3, probs[:, 2].mean(0), atol=0.005))

    def test_bayes_loss_matches_entropy(self):
        # Cross-entropy of the Bayes predictor on its own samples equals its mean entropy.
        h = make("sync_strata")
        rng = np.random.default_rng(2)
        toks, _ = exact.sample(h, 100000, 17, rng)
        eta, probs, _ = exact.beliefs(h, toks)
        ce = -np.log(probs[:, :-1][np.arange(len(toks))[:, None], np.arange(16)[None], toks[:, 1:]]).mean()
        self.assertAlmostEqual(ce, exact.entropy(probs[:, :-1]).mean(), delta=0.005)

    def test_point_stratum_after_s(self):
        # After the rank-1 word s the belief is a single point, whatever the history.
        h = make("sync_strata")
        rng = np.random.default_rng(3)
        toks, _ = exact.sample(h, 20000, 17, rng)
        eta, _, _ = exact.beliefs(h, toks)
        m = toks == 4
        self.assertTrue(m.sum() > 100)
        self.assertLess(eta[m].std(axis=0).max(), 1e-9)

    def test_steps_since(self):
        toks = np.array([[0, 4, 1, 1, 5, 2]])
        self.assertEqual(list(exact.steps_since(toks, {4})[0]), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(list(exact.steps_since(toks, {4, 5})[0]), [-1, 0, 1, 2, 0, 1])


if __name__ == "__main__":
    unittest.main()
