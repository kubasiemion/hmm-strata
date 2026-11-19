"""Exact (analytic) tools for any HMM: sampling, beliefs, Bayes predictor, ground truth."""
from __future__ import annotations
import numpy as np
from .process.base import HMM


def sample(hmm: HMM, n: int, length: int, rng: np.random.Generator):
    """Returns tokens (n, length) and states (n, length): state before emitting token t."""
    V, S = hmm.V, hmm.S
    # joint (token, next_state) distribution per state, flattened
    J = hmm.T.transpose(1, 0, 2).reshape(S, V * S)
    C = np.cumsum(J, axis=1)
    state = rng.choice(S, size=n, p=hmm.pi0)
    toks = np.empty((n, length), dtype=np.int64)
    sts = np.empty((n, length), dtype=np.int64)
    for t in range(length):
        sts[:, t] = state
        u = rng.random(n)[:, None]
        k = (u > C[state]).sum(axis=1)
        k = np.minimum(k, V * S - 1)
        toks[:, t] = k // S
        state = k % S
    return toks, sts


def beliefs(hmm: HMM, tokens: np.ndarray):
    """For each position t: eta[t] = posterior over the hidden state *after* tokens[:t+1]
    (the state before emitting token t+1), and probs[t] = P(token t+1 | tokens[:t+1]).
    Also returns the prior prediction for token 0 as prior_probs (V,)."""
    n, L = tokens.shape
    eta = np.empty((n, L, hmm.S))
    probs = np.empty((n, L, hmm.V))
    emit = hmm.T.sum(axis=2)  # (V, S): P(token | state)
    cur = np.broadcast_to(hmm.pi0, (n, hmm.S)).copy()
    for t in range(L):
        new = np.einsum("ni,nij->nj", cur, hmm.T[tokens[:, t]])
        new /= new.sum(axis=1, keepdims=True)
        eta[:, t] = new
        probs[:, t] = new @ emit.T
        cur = new
    prior_probs = hmm.pi0 @ emit.T
    return eta, probs, prior_probs


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    q = np.where(p > 0, p, 1.0)
    return -(p * np.log(q)).sum(axis=axis)


def tail_mass(p: np.ndarray, k: int) -> np.ndarray:
    """Probability mass outside the top-k tokens."""
    s = np.sort(p, axis=-1)[..., ::-1]
    return 1.0 - s[..., :k].sum(axis=-1)


def steps_since(tokens: np.ndarray, token_set: set[int]) -> np.ndarray:
    """(n, L): steps since the last token in token_set at or before position t (0 = token t
    itself); -1 if none yet."""
    n, L = tokens.shape
    hit = np.isin(tokens, list(token_set))
    out = np.full((n, L), -1, dtype=np.int64)
    last = np.full(n, -10**9)
    for t in range(L):
        last = np.where(hit[:, t], t, last)
        out[:, t] = np.where(last >= 0, t - last, -1)
    return out


def steps_until_state(states: np.ndarray, state_set: set[int], after: int = 1) -> np.ndarray:
    """(n, L): steps from position t until the hidden state at some later position is in
    state_set (state[t+after] ...); -1 if not within the window."""
    n, L = states.shape
    hit = np.isin(states, list(state_set))
    out = np.full((n, L), -1, dtype=np.int64)
    nxt = np.full(n, -1)
    for t in range(L - 1, -1, -1):
        out[:, t] = np.where(nxt >= 0, nxt - t, -1)
        nxt = np.where(hit[:, t], t, nxt)
    return out


def kl(p: np.ndarray, q: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    pp = np.where(p > 0, p, 1.0)
    return (p * (np.log(pp) - np.log(np.maximum(q, eps)))).sum(axis=axis)
