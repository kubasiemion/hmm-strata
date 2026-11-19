"""Token-labelled HMM: the pluggable hidden process."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class HMM:
    """T[x, i, j] = P(emit token x and move to state j | state i).

    sum_x T[x] must be row-stochastic. States are the hidden state *before* the
    token at that position is emitted.
    """
    name: str
    T: np.ndarray                      # (V, S, S)
    pi0: np.ndarray                    # (S,)
    state_names: list[str]
    token_names: list[str]
    state_groups: dict[str, set[int]] = field(default_factory=dict)
    state_events: dict[str, set[int]] = field(default_factory=dict)   # hidden state in set
    token_events: dict[str, set[int]] = field(default_factory=dict)   # emitted token in set
    expected: dict = field(default_factory=dict)  # predictions written before training
    params: dict = field(default_factory=dict)

    @property
    def V(self) -> int:
        return self.T.shape[0]

    @property
    def S(self) -> int:
        return self.T.shape[1]

    def validate(self, tol: float = 1e-9) -> None:
        assert self.T.ndim == 3 and self.T.shape[1] == self.T.shape[2]
        assert (self.T >= -tol).all(), "negative entries"
        rows = self.T.sum(axis=(0, 2))
        assert np.allclose(rows, 1.0, atol=tol), f"rows not stochastic: {rows}"
        assert abs(self.pi0.sum() - 1) < tol and (self.pi0 >= 0).all()
        assert len(self.state_names) == self.S and len(self.token_names) == self.V

    def stationary(self) -> np.ndarray:
        P = self.T.sum(axis=0)  # state-to-state
        w, v = np.linalg.eig(P.T)
        k = np.argmin(np.abs(w - 1))
        pi = np.real(v[:, k])
        pi = np.abs(pi) / np.abs(pi).sum()
        return pi

    def token_ranks(self) -> np.ndarray:
        return np.array([np.linalg.matrix_rank(self.T[x]) for x in range(self.V)])

    def stratum_dims(self) -> np.ndarray:
        """Image dimension of each token's belief map: rank(T[x]) - 1."""
        return self.token_ranks() - 1
