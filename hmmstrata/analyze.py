"""Metrics for one checkpoint: surface (KL to Bayes) and geometry (probe, point stratum)."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from .process import make
from . import exact
from .model import GPT


def load(run: Path, step: int | None = None, dev="cpu"):
    cfg = json.loads((run / "config.json").read_text())
    hmm = make(cfg["process"], **cfg["pparams"])
    ck = sorted(run.glob("ckpt_*.pt"))
    path = ck[-1] if step is None else run / f"ckpt_{step:06d}.pt"
    m = GPT(**cfg["model"]).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev)); m.eval()
    return hmm, m, cfg, path


@torch.no_grad()
def run_model(m, toks, dev, bs=4096):
    P, R = [], []
    for i in range(0, len(toks), bs):
        lg, r = m(torch.from_numpy(toks[i:i + bs]).to(dev), return_resid=True)
        P.append(torch.softmax(lg.float(), -1).cpu().numpy()); R.append(r.float().cpu().numpy())
    return np.concatenate(P), np.concatenate(R)


def ridge_r2(X, Y, seed=0, lam=1e-3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); k = int(0.8 * len(X))
    tr, te = idx[:k], idx[k:]
    Xa = np.hstack([X, np.ones((len(X), 1))])
    A = Xa[tr].T @ Xa[tr] + lam * np.eye(Xa.shape[1])
    W = np.linalg.solve(A, Xa[tr].T @ Y[tr])
    res = Y[te] - Xa[te] @ W
    return 1 - (res ** 2).sum() / ((Y[te] - Y[te].mean(0)) ** 2).sum()


def analyze(run: Path, step=None, n=20000, seed=123, dev="cpu"):
    hmm, m, cfg, path = load(run, step, dev)
    ctx = cfg["ctx"]
    toks, _ = exact.sample(hmm, n, ctx + 1, np.random.default_rng(seed))
    x = toks[:, :ctx]
    eta, bayes, _ = exact.beliefs(hmm, x)
    probs, resid = run_model(m, x, dev)
    kl = exact.kl(bayes, probs)                       # (n, ctx)
    s_tok, f_tok = hmm.token_names.index("s"), hmm.token_names.index("f")
    since_sync = exact.steps_since(x, {s_tok, f_tok})
    since_s = exact.steps_since(x, {s_tok})
    out = {"ckpt": path.name, "excess_kl_nats": float(kl.mean())}
    out["kl_e3_by_since_sync"] = {int(k): float(1e3 * kl[since_sync == k].mean())
                                  for k in (0, 1, 2, 4, 8) if (since_sync == k).any()}
    out["kl_e3_s0"] = float(1e3 * kl[since_s == 0].mean())
    out["kl_e3_f0"] = float(1e3 * kl[(since_sync == 0) & (x == f_tok)].mean())
    pos = np.arange(ctx)[None, :] >= 6
    sel = np.ones_like(kl, bool)
    out["probe_r2_all"] = float(ridge_r2(resid.reshape(-1, resid.shape[-1])[::2], eta.reshape(-1, hmm.S)[::2]))
    # point stratum vs token-matched control (same last token 0, deep after a sync)
    tot_var = lambda A: float(np.trace(np.cov(A.T)))
    g_s = resid[(x == s_tok) & pos]
    g_c = resid[(x == 0) & (since_sync >= 6) & pos]
    out["var_s"], out["var_token0_deep"] = tot_var(g_s), tot_var(g_c)
    out["point_ratio"] = out["var_s"] / out["var_token0_deep"]
    # earlier write-up's convention: centre each group at a fixed position, average over positions 6..15
    def pos_var(mask_fn, hi=15):
        v = []
        for t in range(6, hi):
            A = resid[mask_fn(t), t]
            if len(A) > 30:
                v.append(np.trace(np.cov(A.T)))
        return float(np.mean(v))
    vs = pos_var(lambda t: x[:, t] == s_tok)
    vc = pos_var(lambda t: (x[:, t] == 0) & (since_sync[:, t] >= 5))
    out["point_ratio_perpos_6_14"] = vs / vc          # positions 6..14, as in the earlier write-up
    out["point_ratio_perpos_6_15"] = pos_var(lambda t: x[:, t] == s_tok, ctx) / pos_var(
        lambda t: (x[:, t] == 0) & (since_sync[:, t] >= 5), ctx)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--step", type=int); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    print(json.dumps(analyze(Path(a.run), a.step, dev=a.device), indent=1))
