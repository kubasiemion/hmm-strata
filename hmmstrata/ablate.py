"""Is the surplus pre-sync history load-bearing for the fine surface?

Mean-ablate (replace by its mean) the linear directions that decode the pre-sync token at one residual
stream, and measure KL to Bayes by steps since the last sync.  Controls: random subspaces of the same
dimension (orthogonal to the belief subspace), and the belief subspace itself (positive control).
History directions are made orthogonal to the belief subspace first, so removing them cannot remove the
linearly decodable belief."""
import argparse
from pathlib import Path
import numpy as np, torch
from . import exact
from .analyze import load
from .probes import collect, presync, decode_acc


def orth_basis(W, against=None, tol=1e-6):
    """Orthonormal basis of the column span of W, after removing the span of `against`."""
    if against is not None:
        W = W - against @ (against.T @ W)
    u, s, _ = np.linalg.svd(W, full_matrices=False)
    return u[:, : int((s > tol * s[0]).sum())]


def ridge_cols(X, Y, lam=1e-2):
    Xc = X - X.mean(0)
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (Y - Y.mean(0)))


@torch.no_grad()
def probs_with_ablation(m, x, stream, Q, mu, dev, bs=4096):
    Qt = torch.tensor(Q, dtype=torch.float32, device=dev) if Q is not None else None
    mut = torch.tensor(mu, dtype=torch.float32, device=dev)
    out = []
    for i in range(0, len(x), bs):
        idx = torch.from_numpy(x[i:i + bs]).to(dev)
        h = m.tok(idx) + m.pos(torch.arange(idx.shape[1], device=dev))
        for j in range(len(m.blocks) + 1):
            if j == stream and Qt is not None:
                h = h - ((h - mut) @ Qt) @ Qt.T
            if j < len(m.blocks):
                h = m.blocks[j](h)
        out.append(torch.softmax(m.head(m.lnf(h)).float(), -1).cpu().numpy())
    return np.concatenate(out)


def run(run_dir, streams=(2, 3, 4), n=25000, dev="cpu", n_random=5, ks_fit=(1, 2, 3), seed=0):
    hmm, m, cfg, path = load(run_dir, None, dev)
    ctx, s_tok = cfg["ctx"], hmm.token_names.index("s")
    xf = exact.sample(hmm, n, ctx + 1, np.random.default_rng(5))[0][:, :ctx]
    xe = exact.sample(hmm, n, ctx + 1, np.random.default_rng(123))[0][:, :ctx]
    ef, _, _ = exact.beliefs(hmm, xf)
    ee, bayes, _ = exact.beliefs(hmm, xe)
    lf, _ = collect(m, xf, dev)
    since = exact.steps_since(xe, {s_tok, hmm.token_names.index("f")})
    groups = {"all": np.ones_like(since, bool), "k=0": since == 0, "k=1": since == 1, "k=2": since == 2,
              "k=4": since == 4, "k=8": since == 8}
    rng = np.random.default_rng(seed)

    def kl_groups(p):
        d = exact.kl(bayes, p)
        return {g: 1e3 * float(d[mk].mean()) for g, mk in groups.items()}

    rows = {}
    for stream in streams:
        Rf = lf[stream]                                   # (n, ctx, 64)
        mu = Rf.reshape(-1, Rf.shape[-1]).mean(0).astype(np.float64)
        # belief subspace: linear probe directions
        Xb = Rf.reshape(-1, Rf.shape[-1]).astype(np.float64)
        Wb = ridge_cols(Xb[::2], ef.reshape(-1, hmm.S)[::2])
        Qb = orth_basis(Wb)
        # history directions: decoders of the pre-sync token at several depths
        cols = []
        for k in ks_fit:
            X, y = presync(lf, xf, hmm.V, s_tok, k, stream)
            cols.append(ridge_cols(X.astype(np.float64), np.eye(hmm.V)[y]))
        Qh = orth_basis(np.hstack(cols), against=Qb)
        d = Qh.shape[1]
        base = kl_groups(probs_with_ablation(m, xe, stream, None, mu, dev))
        hist = kl_groups(probs_with_ablation(m, xe, stream, Qh, mu, dev))
        bel = kl_groups(probs_with_ablation(m, xe, stream, Qb, mu, dev))
        rnd = []
        for _ in range(n_random):
            # same regression form as the history decoders, but with the labels shuffled: directions
            # with a matching (low-variance) profile that carry no information about the history
            cols_s = []
            for k in ks_fit:
                X, y = presync(lf, xf, hmm.V, s_tok, k, stream)
                cols_s.append(ridge_cols(X.astype(np.float64), np.eye(hmm.V)[rng.permutation(y)]))
            Qs = orth_basis(np.hstack(cols_s), against=Qb)
            Qs = Qs[:, : d] if Qs.shape[1] >= d else Qs
            rnd.append(kl_groups(probs_with_ablation(m, xe, stream, Qs, mu, dev)))
            var_s = float(np.trace(Qs.T @ np.cov(Xb.T) @ Qs))
        C = np.cov(Xb.T)
        var_h, var_tot = float(np.trace(Qh.T @ C @ Qh)), float(np.trace(C))
        # does a refit linear decoder still read the history after the ablation?  (k=2 accuracy)
        Xa, ya = presync(lf, xf, hmm.V, s_tok, 2, stream)
        Xa = Xa.astype(np.float64)
        Xabl = Xa - (Xa - mu) @ Qh @ Qh.T
        acc_before, base_rate = decode_acc(Xa, ya, hmm.V)
        acc_after = decode_acc(Xabl, ya, hmm.V)[0]
        rows[stream] = dict(dim=d, var_hist=var_h / var_tot, var_ctrl=var_s / var_tot, base=base, hist=hist, belief=bel,
                            random_mean={g: float(np.mean([r[g] for r in rnd])) for g in groups},
                            random_sd={g: float(np.std([r[g] for r in rnd])) for g in groups},
                            dec_before=acc_before, dec_after=acc_after, chance=base_rate)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    for st, r in run(Path(a.run), dev=a.device).items():
        print(f"\nstream {st}: history subspace dim {r['dim']} (variance share {100*r['var_hist']:.2f}%, shuffled control "
              f"{100*r['var_ctrl']:.2f}%); k=2 decodability {r['dec_before']:.2f} -> {r['dec_after']:.2f} (chance {r['chance']:.2f})")
        print("  KL (1e-3):      " + "".join(f"{g:>9s}" for g in r["base"]))
        for lab, key in (("baseline", "base"), ("-history", "hist"), ("-shuffled ctrl", "random_mean"), ("-belief", "belief")):
            print(f"  {lab:14s}  " + "".join(f"{r[key][g]:9.3f}" for g in r["base"]))
