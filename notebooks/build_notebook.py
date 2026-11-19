"""Build notebooks/hmm_strata_baseline.ipynb from the package sources and notebooks/results.json."""
import json
from pathlib import Path
import nbformat as nbf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "notebooks/results.json").read_text())

MODULES = ["hmmstrata/__init__.py", "hmmstrata/device.py", "hmmstrata/exact.py", "hmmstrata/model.py", "hmmstrata/train.py",
           "hmmstrata/analyze.py", "hmmstrata/inherit.py", "hmmstrata/probes.py", "hmmstrata/series.py", "hmmstrata/localdim.py",
           "hmmstrata/split.py", "hmmstrata/calibrate.py", "hmmstrata/process/__init__.py", "hmmstrata/process/base.py",
           "hmmstrata/process/registry.py", "hmmstrata/process/sync_strata.py",
           "tests/test_process.py", "tests/test_inherit.py"]
FILES = {m: (ROOT / m).read_text() for m in MODULES}

# ---------------------------------------------------------------- numbers for the text
fin = R["final"]
def f3(v): return f"{v:.3f}"
rows = "\n".join(
    f"| {s[-1]} | {1e3*fin[s]['excess_kl_nats']:.3f} | {fin[s]['probe_r2_all']:.4f} | {fin[s]['point_ratio_perpos_6_14']:.3f} |"
    for s in sorted(fin))

pr = [fin[s]["point_ratio_perpos_6_14"] for s in fin]
kl_ = [1e3 * fin[s]["excess_kl_nats"] for s in fin]
r2_ = [fin[s]["probe_r2_all"] for s in fin]


def first(o, key, cond):
    for r in o:
        if cond(r[key]):
            return r["step"]


def order_row(name):
    o = R["series"][name]
    h0, h1 = o[0]["hist_k2"], o[-1]["hist_k2"]
    return (first(o, "kl_e3", lambda v: v < 5), first(o, "belief_r2", lambda v: v > 0.99),
            first(o, "kl_e3", lambda v: v < 1), first(o, "hist_k2", lambda v: (v - h0) / (h1 - h0) >= 0.9))


order_lines = "\n".join(f"| {n} | " + " | ".join(str(x) for x in order_row(n)) + " |"
                        for n in ["dest0", "dest1", "dest2", "control0", "control1", "control2", "control3", "control4"])
inh_lines = "\n".join(
    f"| {s[-1]} | " + " | ".join(f"{v[0]-v[1]:+.2f}" for k, v in sorted(R['inherit_final'][s].items(), key=lambda kv: int(kv[0]))) + " |"
    for s in sorted(R["inherit_final"]))

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s, hidden=False: cells.append(
    nbf.v4.new_code_cell(s.strip("\n"), metadata=({"cellView": "form"} if hidden else {})))

md(f"""
# Synchronizing strata

*A small transformer trained on a hidden Markov model with designed synchronizing words: what its activations
represent, how that is learned, and which measurements of it can be trusted.*

This notebook reproduces and checks an earlier write-up of the same setup. It is a baseline: the hidden process
is a plug-in, so the interesting use is to change it and rerun (last sections).

**What is in it**

1. A 4-state, 6-token HMM with a rank-1 word `s` (every history collapses to one belief), a rank-2 word `f`
   (beliefs collapse onto a segment) and four full-rank "fold" tokens.
2. A 143,846-parameter transformer (4 layers, d=64, one 8-dim head per layer) trained on fresh samples.
3. Exact tools: the Bayes belief, the optimal next-token distribution, KL to it, all computed analytically.
4. Measurements: excess KL, a linear belief probe, spread of the point stratum, a history probe,
   learning order across checkpoints, and error inheritance.
5. What did **not** work (a local-dimension estimator), with the calibration that shows why.

**Headline results** (3 seeds, 40k steps; the notebook regenerates a fast single-seed version in minutes)

* The model reaches the Bayes-optimal predictor to within about {max(kl_):.2f} x 10^-3 nats per token, and its final
  residual stream linearly encodes the belief state (R² > {min(r2_)-0.0005:.3f}).
* After the rank-1 word the activations are much tighter than after a fold token (variance ratio {min(pr):.2f} to {max(pr):.2f}),
  but they are not a point: the residual stream keeps pre-sync history, decodable for a few steps.
* The coarse next-token surface is learned before the belief geometry; the last order of magnitude of KL arrives
  after the belief probe has saturated, together with the surplus history.
* Error inheritance (does the model build belief t+1 from its belief at t?) is large at initialisation and mostly
  gone, but not entirely, by the end.

**Caveats** are in the last section. In particular: one HMM, one architecture, three seeds, and all thresholds
in the learning-order analysis are choices.
""")

md("""
## 0. Setup

The package is written by the next cell (collapsed: it is plain Python, about 800 lines). Set `MODE` below.
`fast` trains one seed for 5,000 steps (a few minutes on a Colab GPU); `full` trains three seeds for 40,000
steps (tens of minutes on a GPU, far too slow on CPU). A GPU runtime is recommended. The stored outputs were
produced in fast mode on an Intel Arc GPU (device `xpu`); on Colab the device is `cuda`.
""")

code("#@title Package code (click to expand)\n"
     "import os, sys, pathlib\n"
     f"FILES = {FILES!r}\n"
     "for path, text in FILES.items():\n"
     "    p = pathlib.Path('hmmstrata_pkg') / path\n"
     "    p.parent.mkdir(parents=True, exist_ok=True)\n"
     "    p.write_text(text)\n"
     "sys.path.insert(0, str(pathlib.Path('hmmstrata_pkg').resolve()))\n"
     "print(len(FILES), 'files written')", hidden=True)

code("""
MODE = "fast"  #@param ["fast", "full"]

import json, time
from pathlib import Path
import numpy as np, torch
import matplotlib.pyplot as plt
from hmmstrata.device import pick
from hmmstrata.process import make
from hmmstrata import exact

DEV = pick()
STEPS, SEEDS = (5000, [0]) if MODE == "fast" else (40000, [0, 1, 2])
print("device:", DEV, "| mode:", MODE, "| steps:", STEPS, "| seeds:", SEEDS)
""")

code("#@title Results shipped with the notebook (computed offline on 3 + 5 seeds)\n"
     f"RESULTS = json.loads({json.dumps(json.dumps(R))})\n"
     "print(list(RESULTS))", hidden=True)

md("""
## 1. The process

Each token `x` is a matrix `T[x]`; `T[x][i, j] = P(emit x and move to state j | state i)`. Fold tokens are
emitted from the *destination* state (the convention of the original setup; the other choice gives a different
process and measurably different geometry). The image of a token's belief map has dimension `rank(T[x]) - 1`,
which gives the predictions we check: a point after `s`, a segment after `f`, a fractal after folds.
""")

code("""
hmm = make("sync_strata", dest_emission=True)
print("token ranks:", {k: int(v) for k, v in zip(hmm.token_names, hmm.token_ranks())})
print("predicted stratum dimension (rank-1):", {k: int(v) for k, v in zip(hmm.token_names, hmm.stratum_dims())})
print("stationary distribution:", hmm.stationary().round(3))

# True Bayes beliefs at one position, projected on their two leading principal components.
toks, _ = exact.sample(hmm, 20000, 16, np.random.default_rng(0))
eta, probs, _ = exact.beliefs(hmm, toks)
t = 12
B = eta[:, t]
C = B - B.mean(0)
u, s, vt = np.linalg.svd(C, full_matrices=False)
P2 = C @ vt[:2].T
last = toks[:, t]
plt.figure(figsize=(5, 4.5))
for name, tok, col in [("ends in fold token", None, "#999"), ("ends in s (point)", 4, "#d33"), ("ends in f (segment)", 5, "#e90")]:
    m = (last < 4) if tok is None else (last == tok)
    plt.scatter(P2[m, 0], P2[m, 1], s=3, alpha=0.4 if tok is None else 0.9, c=col, label=name)
plt.legend(markerscale=4); plt.title("True belief states (position 12)"); plt.axis("equal"); plt.show()
""")

md("""
## 2. Unit tests of the exact tools

Sampling frequencies match the analytic predictions, the Bayes predictor's cross-entropy equals its own
entropy, the belief after `s` is a single point, and the error-inheritance Jacobian matches a finite
difference and separates a synthetic recursive estimator from a parallel one.
""")

code("""
import unittest
suite = unittest.defaultTestLoader.discover("hmmstrata_pkg/tests", pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=1).run(suite)
assert result.wasSuccessful()
""")

md("""
## 3. Train

Fresh sequences every step (no fixed dataset), context 16, batch 128, AdamW lr 1e-3, weight decay 0, cosine
decay to 0, no warmup. Checkpoints are log-spaced from step 0.
""")

code("""
from hmmstrata.train import train
RUNS = {}
for seed in SEEDS:
    out = Path(f"runs/seed{seed}")
    t0 = time.time()
    n_params = train("sync_strata", {"dest_emission": True}, seed, STEPS, out, DEV, wd=0.0, n_ckpt=24)
    RUNS[seed] = out
    print(f"seed {seed}: {n_params:,} parameters, {time.time() - t0:.0f} s")
""")

md(f"""
## 4. Surface and geometry at the final checkpoint

* **Excess KL**: `KL(Bayes || model)` averaged over positions, in 10^-3 nats.
* **Probe R²**: a linear map from the final residual stream to the true belief, on held-out sequences.
* **Point ratio**: total variance of the residual vectors at contexts ending in `s`, divided by the same for
  contexts ending in fold token 0 at least 5 steps after a sync (the control that holds the last token fixed),
  each centred per position and averaged over positions 6 to 14.

Shipped results for 3 seeds (this configuration, 40k steps):

| seed | excess KL (1e-3) | probe R² | point ratio |
|---|---|---|---|
{rows}

Fast mode (5,000 steps) is not converged: expect an excess KL of roughly 0.4 x 10^-3 and a looser point ratio of
about 0.1 to 0.15. Use `full` to match the table.

The earlier write-up reported a point ratio of 0.056 and an excess KL of about 0.08 x 10^-3 after correcting a
bug of its own (an untrained position in its analysis). Emitting fold tokens from the source state instead of
the destination state gave a ratio of 0.09 to 0.20, so the emission convention matters.
""")

code("""
from hmmstrata.analyze import analyze
live = {}
for seed, run in RUNS.items():
    a = analyze(run, dev=DEV)
    live[seed] = a
    print(f"seed {seed}: excess KL {1e3*a['excess_kl_nats']:.3f}e-3 | probe R2 {a['probe_r2_all']:.4f} | "
          f"point ratio (pos 6-14) {a['point_ratio_perpos_6_14']:.3f}")
    print("   KL (1e-3) by steps since the last sync:", {k: round(v, 3) for k, v in a["kl_e3_by_since_sync"].items()})
""")

md("""
### Surplus history

The belief after `s` is a single point, so the token *before* the `s` carries no information about the future:
any decoding of it above the majority baseline is history the model keeps but does not need. The probe decodes
that token from the residual stream `k` steps after the `s`, at the final layer and at each layer for `k = 2`.
""")

code("""
from hmmstrata.probes import evaluate
for seed, run in RUNS.items():
    r = evaluate(run, dev=DEV)
    print(f"seed {seed}: belief R2 by stream (embedding, layers 1-4):", " ".join(f"{v:.3f}" for v in r["belief_r2"]))
    print("  pre-sync token decoding, final layer (accuracy / majority baseline):")
    for k, (a, b, n) in r["presync_final"].items():
        print(f"    {k} steps after s: {a:.2f} / {b:.2f}")
    print("  k=2 by stream (embedding, L1..L4):", " ".join(f"{a:.2f}" for a, b in r["presync_layers"]))
""")

md(f"""
## 5. Learning order across training

For every checkpoint we measure KL, the belief-probe R² and the history decodable at `k = 2`. The answer to
"what is learned first" depends on the threshold, so the table gives the first checkpoint reaching fixed levels
(the grid is log-spaced and coarse):

| run | KL < 5e-3 | R² > 0.99 | KL < 1e-3 | history k=2 at 90% of its gain |
|---|---|---|---|---|
{order_lines}

`dest*` are the destination-emission runs of this notebook, `control*` an earlier variant (source-state emission,
weight decay 0.01). Reading: the coarse surface comes first; the last order of magnitude of KL arrives after the
belief probe has saturated and together with the surplus history. Measured instead as "90% of the total KL
improvement on a linear scale", KL looks finished by step 16 to 40, which is an artefact of its huge starting
value and should not be quoted as "the surface settles first".
""")

code("""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for name, o in RESULTS["series"].items():
    st = np.array([r["step"] for r in o]) + 1
    col = "C0" if name.startswith("dest") else "C1"
    ax[0].plot(st, [r["kl_e3"] for r in o], c=col, alpha=.7)
    ax[1].plot(st, [r["belief_r2"] for r in o], c=col, alpha=.7)
    ax[2].plot(st, [r["hist_k2"] for r in o], c=col, alpha=.7)
ax[0].set_yscale("log"); ax[0].set_title("KL to Bayes (1e-3 nats)")
ax[1].set_title("belief-probe R2"); ax[2].set_title("pre-sync token decodable, k=2")
for a in ax: a.set_xscale("log"); a.set_xlabel("training step")
plt.tight_layout(); plt.show()
print("blue: destination-emission seeds (this notebook's setup), orange: earlier variant")
""")

code("""
# Optional: the same series for the model(s) trained above (takes a few minutes).
RUN_SERIES = False
if RUN_SERIES:
    from hmmstrata.series import series
    for seed, run in RUNS.items():
        o = series(run, n=8000, dev=DEV)
        print(seed, [(r["step"], round(r["kl_e3"], 3), round(r["belief_r2"], 3), round(r["hist_k2"], 2)) for r in o][::4])
""")

md(f"""
## 6. Error inheritance

If a model folds recursively, so that its belief at `t+1` is the true update applied to its belief at `t`, then
its error at `t` is carried to `t+1` through the Jacobian of the update. If it computes each position afresh
from the context, the errors are unrelated. For pairs of positions at depth `k` after an `s`, followed by a fold
token, the score is the correlation between the probe error at `t+1` and the probe error at `t` pushed through
the true Jacobian, minus the same correlation with the `t+1` errors shuffled among the pairs.

Shipped final-checkpoint values (real minus shuffled; depths 1 to 6):

| seed | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|
{inh_lines}

At initialisation the score is about +0.4 to +0.5 at depths 3 to 5. It decays slowly during training. The final
models compute each position mostly afresh, with a weak residual inheritance on some seeds. Depth 1 is negative
on most seeds and unexplained.
""")

code("""
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for j, k in enumerate(("3", "5")):
    for s, d in RESULTS["inherit_dev"].items():
        st = sorted(int(x) for x in d)
        ax[j].plot(np.array(st) + 1, [d[str(x)][k][0] - d[str(x)][k][1] for x in st], marker="o", ms=3, label=s)
    ax[j].set_xscale("log"); ax[j].axhline(0, c="k", lw=.5); ax[j].set_title(f"inheritance, depth {k} (real - shuffled)")
    ax[j].set_xlabel("training step")
ax[0].legend(); plt.tight_layout(); plt.show()
""")

code("""
from hmmstrata.inherit import inheritance
for seed, run in RUNS.items():
    r = inheritance(run, dev=DEV, n=30000)
    print(f"seed {seed}: real - shuffled by depth:", {k: round(v[0] - v[1], 3) for k, v in r.items()})
""")

md("""
## 7. What did not work: local dimension of the residual stream

The natural way to look for "strata" in activations is a local-dimension estimator. We tried TwoNN and a
correlation-dimension slope on the residual stream. They did **not** discriminate the strata. The calibration
below uses synthetic sets of known dimension (0 to 3) with noise in 5 extra dimensions. TwoNN reads 7 to 9
whatever the true dimension, and the slopes drop to zero beyond the extent of the set, so a point and a short
segment look alike. Conclusions drawn from those estimators on the real activations would not be safe, and the
true belief geometry (computed from the HMM) is the reliable reference.
""")

code("""
from hmmstrata.calibrate import synthetic
synthetic()
""")

md("""
## 8. Make the hidden state your own

A process is a token-labelled HMM: `T[x, i, j]` with `sum_x T[x]` row-stochastic, plus optional annotations
(named state groups, token and state events, an `expected` dictionary for predictions written down before
training). Register it by name and everything above runs unchanged. The example below makes the fold part
asymmetric (a different stay probability per state, keeping `s` and `f`). It is only constructed and checked
here; training and analysing it is the exercise.
""")

code("""
import numpy as np
from hmmstrata.process import register, make
from hmmstrata.process.base import HMM

@register("sync_strata_asym")
def sync_strata_asym(stays=(0.95, 0.85, 0.7, 0.5), match=0.70, p_s=0.08, p_f=0.12):
    S, F = 4, 4
    V = F + 2
    p_fold = 1 - p_s - p_f
    M = np.zeros((S, S))
    for i, st in enumerate(stays):
        M[i] = (1 - st) / (S - 1); M[i, i] = st
    E = np.full((S, F), (1 - match) / (F - 1)); np.fill_diagonal(E, match)
    T = np.zeros((V, S, S))
    for x in range(F):
        T[x] = p_fold * M * E[:, x][None, :]            # destination emission
    T[F] = p_s * np.outer(np.ones(S), np.eye(S)[0])      # s: any state -> state 0 (rank 1)
    T[F + 1] = p_f * np.array([[0, .8, .2, 0]] * 2 + [[0, .2, .8, 0]] * 2)   # f: collapse to a segment (rank 2)
    h = HMM(name="sync_strata_asym", T=T, pi0=np.full(S, 1 / S), state_names=[f"q{i}" for i in range(S)],
            token_names=[str(i) for i in range(F)] + ["s", "f"], token_events={"s": {F}, "f": {F + 1}})
    h.pi0 = h.stationary()
    return h

h2 = make("sync_strata_asym")
print("ranks:", {k: int(v) for k, v in zip(h2.token_names, h2.token_ranks())})
print("stationary distribution:", h2.stationary().round(3), "(every state should keep non-trivial mass)")
# To train and analyse it:  train("sync_strata_asym", {}, seed, STEPS, Path("runs/asym"), DEV, wd=0.0)
# (hmmstrata.analyze looks up token names 's' and 'f', so it works for any process that keeps them.)
""")

md(f"""
## 9. Limits

* One HMM, one architecture and context length, three seeds in the shipped results. Seeds disagree noticeably
  on the point ratio ({min(pr):.2f} to {max(pr):.2f}) and on how much history is retained.
* Probe-based measurements are only as good as the linear probe; the error-inheritance score goes through one.
* Every threshold in the learning-order table is a choice. The checkpoint grid is coarse.
* The point-stratum variance ratio measures spread, not dimension.
* The earlier variant (source-state emission) and this one differ, so numbers from the two must not be mixed.

## References

* Shai, Marzen, Teixeira, Gietelink Oldenziel, Riechers (2024). *Transformers represent belief state geometry in
  their residual stream.* https://arxiv.org/abs/2405.15943
* Piotrowski, Riechers, Filan, Shai (2025). *Constrained belief updates explain geometric structures in
  transformer representations.* https://arxiv.org/abs/2502.01954
* Robinson, Dey, Chiang (2025). *Token embeddings violate the manifold hypothesis.* https://arxiv.org/abs/2504.01002
* Robinson, Dey, Chiang (2024). *The structure of the token space for large language models.* https://arxiv.org/abs/2410.08993
* Wang et al. (2025). *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement
  Learning for LLM Reasoning.* https://arxiv.org/abs/2506.01939

The setup follows an earlier private write-up produced with a Claude model; the destination-emission HMM, the
point-ratio definition and the surplus-history idea come from it. The checks here (position-15 artefact in the
original analysis, emission convention, learning order, inheritance, the failed dimension estimators) are ours.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "colab": {"provenance": [], "toc_visible": True, "name": "hmm_strata_baseline.ipynb"},
    "accelerator": "GPU"})
nbf.write(nb, ROOT / "notebooks/hmm_strata_baseline.ipynb")
print("written", ROOT / "notebooks/hmm_strata_baseline.ipynb", len(cells), "cells")
