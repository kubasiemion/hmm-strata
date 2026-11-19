"""Calibrate the dimension estimators on synthetic sets of known dimension and noise,
then put subsampling error bars on the real slopes."""
import numpy as np
from pathlib import Path
from .localdim import twonn, mle_dim, groups
from .split import corr_slopes, probe_basis
from . import exact
from .analyze import load, run_model

MULTS = (2, 4, 8, 16, 32)


def synth(d, L, sigma, noise_dims=5, n=15000, D=64, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.random((n, d)) * L if d else np.zeros((n, 0))
    Qm, _ = np.linalg.qr(rng.standard_normal((D, d + noise_dims)))
    X = np.zeros((n, d + noise_dims))
    X[:, :d] = lat
    X[:, d:] = rng.standard_normal((n, noise_dims)) * sigma
    return X @ Qm.T


def fmt(A):
    sl = [corr_slopes(A, MULTS, seed=s) for s in range(8)]
    m, sd = np.mean(sl, 0), np.std(sl, 0)
    return " ".join(f"{a:4.2f}±{b:.2f}" for a, b in zip(m, sd))


def synthetic():
    print("SYNTHETIC (noise: sigma=1 in 5 dims). cols: TwoNN MLE | slopes at 2,4,8,16,32 x NN")
    for d, L in [(0, 1), (1, 20), (1, 100), (2, 20), (2, 100), (3, 100)]:
        A = synth(d, L, 1.0)
        print(f"true d={d} L={L:3d}  {twonn(A)[0]:5.2f} {mle_dim(A):5.2f} | {fmt(A)}")


def real(run_dir, t=12, n=200000, dev="xpu"):
    hmm, m, cfg, path = load(run_dir, None, dev)
    xt = exact.sample(hmm, 20000, cfg["ctx"], np.random.default_rng(11))[0]
    et, _, _ = exact.beliefs(hmm, xt); _, rt = run_model(m, xt, dev)
    mu, Q = probe_basis(rt, et)
    x = exact.sample(hmm, n, cfg["ctx"], np.random.default_rng(7))[0]
    eta, _, _ = exact.beliefs(hmm, x); _, resid = run_model(m, x, dev)
    print(f"\n{run_dir.name} (mean±sd over 8 subsamples)")
    for name, mask in groups(x, hmm, t).items():
        if name not in ("ends in s (rank 1)", "ends in f (rank 2)", "token 0, >=6 after any sync"):
            continue
        R = resid[mask, t].astype(np.float64) - mu
        for sp, A in (("full", R), ("aligned", R @ Q)):
            print(f"{name:28s} {sp:8s} {fmt(A)}")


if __name__ == "__main__":
    synthetic()
    for s in (0, 1, 2):
        real(Path(f"runs/control_sync_strata/s{s}"))
