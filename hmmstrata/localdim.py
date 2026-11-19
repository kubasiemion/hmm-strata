"""Local intrinsic dimension (TwoNN and Levina-Bickel MLE) per context group, for the model's
residual stream and for the true belief vectors on the same contexts."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from . import exact
from .analyze import load, run_model


def _dedup(X, decimals=7):
    _, i = np.unique(np.round(X, decimals), axis=0, return_index=True)
    return X[np.sort(i)]


def twonn(X, max_pts=20000, trim=0.1, seed=0):
    """Facco et al. 2017: d = N / sum log(r2/r1), top `trim` fraction of ratios discarded."""
    X = _dedup(X)
    if len(X) < 50:
        return float("nan"), len(X)
    if len(X) > max_pts:
        X = X[np.random.default_rng(seed).choice(len(X), max_pts, replace=False)]
    d, _ = cKDTree(X).query(X, k=3)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    mu = mu[: int(len(mu) * (1 - trim))]
    return float(len(mu) / np.log(mu).sum()), len(X)


def mle_dim(X, k=10, max_pts=20000, seed=0):
    """Levina-Bickel 2004 MLE with k neighbours, averaged over points."""
    X = _dedup(X)
    if len(X) < 2 * k:
        return float("nan")
    if len(X) > max_pts:
        X = X[np.random.default_rng(seed).choice(len(X), max_pts, replace=False)]
    d, _ = cKDTree(X).query(X, k=k + 1)
    d = d[:, 1:]
    d = d[d[:, 0] > 0]
    inv = np.log(d[:, -1:] / d[:, :-1]).sum(1) / (k - 1)
    return float(1.0 / inv.mean())


def groups(x, hmm, t):
    s, f = hmm.token_names.index("s"), hmm.token_names.index("f")
    sync = exact.steps_since(x, {s, f})
    since_s = exact.steps_since(x, {s})
    col = x[:, t]
    g = {"ends in s (rank 1)": col == s, "ends in f (rank 2)": col == f}
    for k in (1, 2, 3):
        g[f"token 0, {k} after s"] = (col == 0) & (since_s[:, t] == k)
    g["token 0, >=6 after any sync"] = (col == 0) & (sync[:, t] >= 6)
    return g


def run(run_dir: Path, step=None, n=200000, t=12, seed=7, dev="xpu"):
    hmm, m, cfg, path = load(run_dir, step, dev)
    x = exact.sample(hmm, n, cfg["ctx"], np.random.default_rng(seed))[0]
    eta, _, _ = exact.beliefs(hmm, x)
    _, resid = run_model(m, x, dev)
    rows = []
    for name, mask in groups(x, hmm, t).items():
        A, B = resid[mask, t].astype(np.float64), eta[mask, t]
        da, na = twonn(A); db, nb = twonn(B)
        rows.append(dict(group=name, n=int(mask.sum()), n_unique_belief=int(len(_dedup(B))),
                         act_twonn=da, act_mle=mle_dim(A), belief_twonn=db, belief_mle=mle_dim(B)))
    return dict(ckpt=path.name, position=t, rows=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--step", type=int); ap.add_argument("--device", default="xpu")
    a = ap.parse_args()
    r = run(Path(a.run), a.step, dev=a.device)
    print(r["ckpt"], "position", r["position"])
    print(f"{'group':32s} {'n':>6s} {'uniq_b':>6s} | {'act TwoNN':>9s} {'act MLE':>8s} | {'bel TwoNN':>9s} {'bel MLE':>8s}")
    for w in r["rows"]:
        print(f"{w['group']:32s} {w['n']:6d} {w['n_unique_belief']:6d} | {w['act_twonn']:9.2f} {w['act_mle']:8.2f} | {w['belief_twonn']:9.2f} {w['belief_mle']:8.2f}")
