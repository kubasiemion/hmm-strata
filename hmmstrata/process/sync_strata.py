"""(dest_emission=True follows the methods sketch appended to the write-up.)
The 4-state, 6-token HMM from the earlier 'Synchronizing Strata' write-up.

Reconstructed from the write-up's prose; where it is ambiguous the choice is
stated here:
  * every state emits s with prob p_s, f with prob p_f, a fold token otherwise;
  * fold token x from state i: T[x][i, j] = p_fold * E[i, x] * M[i, j], where
    M stays with prob `stay` and otherwise is uniform over the other states,
    and E[i, x] is `match` for x == i and uniform over the other fold tokens;
  * s (rank 1): jumps to state 0 from anywhere;
  * f (rank 2): moves to (0,.8,.2,0) from states {0,1} and (0,.2,.8,0) from {2,3}.
"""
import numpy as np
from .base import HMM
from .registry import register


@register("sync_strata")
def sync_strata(stay=0.82, match=0.70, p_s=0.08, p_f=0.12, dest_emission=False) -> HMM:
    S, F = 4, 4
    V = F + 2
    S_TOK, F_TOK = F, F + 1
    p_fold = 1.0 - p_s - p_f
    M = np.full((S, S), (1 - stay) / (S - 1))
    np.fill_diagonal(M, stay)
    E = np.full((S, F), (1 - match) / (F - 1))
    np.fill_diagonal(E, match)
    T = np.zeros((V, S, S))
    for x in range(F):
        if dest_emission:   # earlier write-up's methods sketch: T[x] = p_fold * P @ diag(Q[:, x]); Q[j, x] = E[j, x]
            T[x] = p_fold * M * E[:, x][None, :]
        else:               # first reconstruction: emission from the source state
            T[x] = p_fold * E[:, [x]] * M
    T[S_TOK] = p_s * np.outer(np.ones(S), np.eye(S)[0])
    frows = np.array([[0, .8, .2, 0], [0, .8, .2, 0], [0, .2, .8, 0], [0, .2, .8, 0]])
    T[F_TOK] = p_f * frows
    hmm = HMM(
        name="sync_strata", T=T, pi0=np.full(S, 1 / S),
        state_names=[f"q{i}" for i in range(S)],
        token_names=[str(i) for i in range(F)] + ["s", "f"],
        token_events={"s": {S_TOK}, "f": {F_TOK}, "sync": {S_TOK, F_TOK}},
        expected={"stratum_dims": {"s": 0, "f": 1, "fold": 3},
                  "note": "point stratum after s; segment after f; fold tokens full rank"},
    )
    hmm.pi0 = hmm.stationary()
    return hmm
