"""Error inheritance: does the model's belief error at t+1 carry the error at t, pushed through
the true update map?  A model folding recursively (belief_{t+1} = f_x(belief_t)) passes errors
along: corr ~ 1.  A model computing every position afresh from the context: corr ~ 0.

Pairs (t, t+1) with the last sync an `s` at depth k (steps since s at t, k = 1..K) and a fold
token x at t+1.  e_t = probe_t - eta_t.  propagated = e_t @ J_x(eta_t) with the Jacobian of the
true update f_x(eta) = eta T_x / (eta T_x 1):   J[i,j] = (T_x[i,j] - r_i f_j) / s,  r = T_x 1.
Score = Pearson correlation between propagated and e_{t+1} (all components pooled), against a
control that shuffles e_{t+1} among the pairs of the same depth."""
import argparse
from pathlib import Path
import numpy as np
from . import exact
from .analyze import load, run_model


def jacobian_push(hmm, eta, x_next, e):
    Tx = hmm.T[x_next]                                   # (n, S, S)
    u = np.einsum("ni,nij->nj", eta, Tx)
    s = u.sum(1)
    f = u / s[:, None]
    r = Tx.sum(2)
    J = (Tx - r[:, :, None] * f[:, None, :]) / s[:, None, None]
    return np.einsum("ni,nij->nj", e, J)


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def inheritance_from_estimates(hmm, x, eta, est, depths=range(1, 7), seed=0):
    """x (n,L) tokens, eta (n,L,S) true beliefs, est (n,L,S) estimated beliefs."""
    rng = np.random.default_rng(seed)
    s_tok = hmm.token_names.index("s")
    sync = {hmm.token_names.index(t) for t in ("s", "f")}
    since_s = exact.steps_since(x, {s_tok})
    since_any = exact.steps_since(x, sync)
    fold = x < s_tok
    out = {}
    for k in depths:
        sel = (since_s[:, :-1] == k) & (since_any[:, :-1] == k) & fold[:, 1:]
        ni, ti = np.nonzero(sel)
        if len(ni) < 200:
            continue
        e_t = est[ni, ti] - eta[ni, ti]
        e_n = est[ni, ti + 1] - eta[ni, ti + 1]
        prop = jacobian_push(hmm, eta[ni, ti], x[ni, ti + 1], e_t)
        out[k] = (corr(prop, e_n), corr(prop, e_n[rng.permutation(len(e_n))]), len(ni))
    return out


def fit_probe(R, eta, lam=1e-3):
    X = np.hstack([R.reshape(-1, R.shape[-1]), np.ones((R.shape[0] * R.shape[1], 1))]).astype(np.float64)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ eta.reshape(-1, eta.shape[-1]))


def apply_probe(W, R):
    X = np.concatenate([R, np.ones(R.shape[:-1] + (1,))], -1).astype(np.float64)
    return X @ W


def inheritance(run_dir: Path, step=None, n=40000, dev="cpu", depths=range(1, 7)):
    hmm, m, cfg, path = load(run_dir, step, dev)
    ctx = cfg["ctx"]
    xf = exact.sample(hmm, n, ctx, np.random.default_rng(11))[0]
    ef, _, _ = exact.beliefs(hmm, xf)
    _, rf = run_model(m, xf, dev)
    W = fit_probe(rf, ef)
    x = exact.sample(hmm, n, ctx, np.random.default_rng(7))[0]
    eta, _, _ = exact.beliefs(hmm, x)
    _, r = run_model(m, x, dev)
    return inheritance_from_estimates(hmm, x, eta, apply_probe(W, r), depths)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--step", type=int); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    res = inheritance(Path(a.run), a.step, dev=a.device)
    print("depth  corr(real)  corr(shuffled)  n")
    for k, (c, cs, nn) in res.items():
        print(f"{k:5d}  {c:9.3f}  {cs:12.3f}  {nn}")
