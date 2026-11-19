"""Extended probes: belief R2 and pre-sync-token decoding, across layers, steps-since-s, and
checkpoints. The true belief after s is a single point, so the pre-sync token carries no
information about the future: any decoding above the majority baseline is surplus history."""
import json, sys
from pathlib import Path
import numpy as np, torch
from . import exact
from .analyze import load, ridge_r2


@torch.no_grad()
def collect(m, x, dev, bs=4096):
    out = None
    for i in range(0, len(x), bs):
        lg, rs = m(torch.from_numpy(x[i:i + bs]).to(dev), return_resid="all")
        rs = [r.float().cpu().numpy() for r in rs]
        P = torch.softmax(lg.float(), -1).cpu().numpy()
        if out is None:
            out = ([[] for _ in rs], [])
        for j, r in enumerate(rs):
            out[0][j].append(r)
        out[1].append(P)
    return [np.concatenate(a) for a in out[0]], np.concatenate(out[1])


def decode_acc(X, y, V, lam=1e-2, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X)); k = int(0.8 * len(X))
    tr, te = idx[:k], idx[k:]
    Xa = np.hstack([X, np.ones((len(X), 1))]).astype(np.float64)
    Y = np.eye(V)[y]
    W = np.linalg.solve(Xa[tr].T @ Xa[tr] + lam * np.eye(Xa.shape[1]), Xa[tr].T @ Y[tr])
    acc = (np.argmax(Xa[te] @ W, 1) == y[te]).mean()
    base = (y[te] == np.bincount(y[tr], minlength=V).argmax()).mean()
    return float(acc), float(base)


def presync(layers, x, V, s_tok, k, layer):
    n, L = x.shape
    since = exact.steps_since(x, {s_tok})
    mask = (since == k) & (np.arange(L)[None, :] >= k + 1)
    ni, ti = np.nonzero(mask)
    return layers[layer][ni, ti], x[ni, ti - k - 1]


def evaluate(run_dir, step=None, n=25000, dev="xpu", ks=(1, 2, 3, 4, 6, 8), layers_k=2):
    hmm, m, cfg, path = load(run_dir, step, dev)
    x = exact.sample(hmm, n, cfg["ctx"] + 1, np.random.default_rng(5))[0][:, :cfg["ctx"]]
    eta, bayes, _ = exact.beliefs(hmm, x)
    layers, probs = collect(m, x, dev)
    s_tok = hmm.token_names.index("s")
    res = dict(ckpt=path.name, kl_e3=float(1e3 * exact.kl(bayes, probs).mean()))
    res["belief_r2"] = [float(ridge_r2(l.reshape(-1, l.shape[-1])[::3], eta.reshape(-1, hmm.S)[::3])) for l in layers]
    res["presync_final"] = {}
    for k in ks:
        X, y = presync(layers, x, hmm.V, s_tok, k, -1)
        res["presync_final"][k] = decode_acc(X, y, hmm.V) + (len(y),)
    res["presync_layers"] = [decode_acc(*presync(layers, x, hmm.V, s_tok, layers_k, l), hmm.V) for l in range(len(layers))]
    return res


if __name__ == "__main__":
    for s in (0, 1, 2):
        r = evaluate(Path(f"runs/control_sync_strata/s{s}"))
        print(f"\nseed {s} final: belief R2 by stream (emb,L1..L4): " + " ".join(f"{v:.3f}" for v in r["belief_r2"]))
        print("  pre-sync token decode at final layer (acc / majority baseline, n):")
        for k, (a, b, nn) in r["presync_final"].items():
            print(f"    k={k}: {a:.3f} / {b:.3f}  (n={nn})")
        print("  k=2 by stream (emb,L1..L4): " + " ".join(f"{a:.2f}/{b:.2f}" for a, b in r["presync_layers"]))
