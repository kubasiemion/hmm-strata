"""Split the residual stream into the belief-aligned subspace (span of the linear probe's
directions) and its orthogonal complement; local dimension of each, per context group,
plus a scale-resolved correlation-dimension slope."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from . import exact
from .analyze import load, run_model
from .localdim import twonn, mle_dim, groups, _dedup


def probe_basis(resid, eta, lam=1e-3):
    X = resid.reshape(-1, resid.shape[-1]).astype(np.float64)
    Y = eta.reshape(-1, eta.shape[-1])
    mu = X.mean(0)
    Xc = X - mu
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (Y - Y.mean(0)))  # (d, S)
    u, s, _ = np.linalg.svd(W, full_matrices=False)
    k = int((s > 1e-6 * s[0]).sum())
    return mu, u[:, :k]          # orthonormal basis of the probe's directions (rank <= S-1)


def corr_slopes(X, mults=(2, 4, 8, 16, 32), max_pts=4000, seed=0):
    X = _dedup(X)
    if len(X) < 200:
        return [float("nan")] * len(mults)
    if len(X) > max_pts:
        X = X[np.random.default_rng(seed).choice(len(X), max_pts, replace=False)]
    tree = cKDTree(X)
    nn = np.median(tree.query(X, k=2)[0][:, 1])
    r = nn * np.array(mults, float)
    C = tree.count_neighbors(tree, r).astype(float) - len(X)
    lo = tree.count_neighbors(tree, r / 1.5).astype(float) - len(X)
    return [float(np.log(C[i] / max(lo[i], 1)) / np.log(1.5)) for i in range(len(r))]


def run(run_dir: Path, step=None, n=200000, t=12, seed=7, dev="xpu"):
    hmm, m, cfg, path = load(run_dir, step, dev)
    # probe fitted on a separate sample (all positions), so the split is not tuned to the groups
    xt = exact.sample(hmm, 20000, cfg["ctx"], np.random.default_rng(11))[0]
    et, _, _ = exact.beliefs(hmm, xt)
    _, rt = run_model(m, xt, dev)
    mu, Q = probe_basis(rt, et)
    x = exact.sample(hmm, n, cfg["ctx"], np.random.default_rng(seed))[0]
    eta, _, _ = exact.beliefs(hmm, x)
    _, resid = run_model(m, x, dev)
    sel = {k: v for k, v in groups(x, hmm, t).items()
           if k in ("ends in s (rank 1)", "ends in f (rank 2)", "token 0, >=6 after any sync")}
    print(f"{path.parent.name}/{path.name}  probe rank {Q.shape[1]}  position {t}")
    print(f"{'group':30s} {'space':10s} {'TwoNN':>6s} {'MLE':>6s} | corr-dim slope at 2,4,8,16,32 x NN-scale")
    for name, mask in sel.items():
        R = resid[mask, t].astype(np.float64) - mu
        parts = {"aligned": R @ Q, "orthogonal": R - (R @ Q) @ Q.T, "full": R, "belief": eta[mask, t]}
        for sp, A in parts.items():
            d, _ = twonn(A)
            sl = corr_slopes(A)
            print(f"{name:30s} {sp:10s} {d:6.2f} {mle_dim(A):6.2f} | " + " ".join(f"{v:5.2f}" for v in sl))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--step", type=int); ap.add_argument("--device", default="xpu")
    a = ap.parse_args()
    run(Path(a.run), a.step, dev=a.device)
