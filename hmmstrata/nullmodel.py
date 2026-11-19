"""Shuffled-residual null: activations = linear map of the TRUE belief + residual noise taken
from other contexts of the same group. Same estimators on null and real, group by group."""
import numpy as np
from pathlib import Path
from . import exact
from .analyze import load, run_model
from .localdim import groups
from .split import corr_slopes

M = (2, 4, 8, 16, 32)
KEEP = ("ends in f (rank 2)", "token 0, >=6 after any sync", "token 0, 3 after s")


def fit_map(eta, R):
    Xa = np.hstack([eta, np.ones((len(eta), 1))])
    return np.linalg.lstsq(Xa, R, rcond=None)[0]


def run(run_dir, t=12, dev="xpu"):
    hmm, m, cfg, path = load(run_dir, None, dev)
    def data(seed, n):
        x = exact.sample(hmm, n, cfg["ctx"], np.random.default_rng(seed))[0]
        eta, _, _ = exact.beliefs(hmm, x); _, r = run_model(m, x, dev)
        return x, eta[:, t], r[:, t].astype(np.float64)
    xf, ef, rf = data(11, 200000)
    W = fit_map(ef, rf)
    xe, ee, re_ = data(7, 200000)
    gf, ge = groups(xf, hmm, t), groups(xe, hmm, t)
    rng = np.random.default_rng(0)
    print(f"\n{run_dir.name}: slopes at 2,4,8,16,32 x NN (mean of 6 subsamples); linear map R2 = "
          f"{1 - ((rf - np.hstack([ef, np.ones((len(ef),1))]) @ W)**2).sum() / ((rf - rf.mean(0))**2).sum():.3f}")
    for name in ge:
        if name not in KEEP:
            continue
        me, mf = ge[name], gf[name]
        lin = np.hstack([ee[me], np.ones((me.sum(), 1))]) @ W
        resid_pool = (rf[mf] - np.hstack([ef[mf], np.ones((mf.sum(), 1))]) @ W)
        null = lin + resid_pool[rng.integers(0, len(resid_pool), me.sum())]
        for tag, A in (("real", re_[me]), ("null", null)):
            sl = np.mean([corr_slopes(A, M, seed=s) for s in range(6)], 0)
            print(f"{name:28s} {tag}  " + " ".join(f"{v:5.2f}" for v in sl))


if __name__ == "__main__":
    for s in (0, 1, 2):
        run(Path(f"runs/control_sync_strata/s{s}"))
