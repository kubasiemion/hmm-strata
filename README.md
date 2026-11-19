# hmm-strata

Belief-state geometry in a small transformer trained on a hidden Markov model with designed **synchronizing
words**: what its activations represent, how that is learned, and which measurements of it can be trusted.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kubasiemion/hmm-strata/blob/main/notebooks/hmm_strata_baseline.ipynb)

The notebook is the quickest way in: it writes the package, trains a model (about 2 minutes in fast mode on a GPU),
and reproduces every number below. The hidden process is a plug-in, so the intended use is to change it and rerun.

## The setup

- **Process.** A 4-state, 6-token HMM. Four full-rank "fold" tokens (emitted from the destination state), a
  rank-1 word `s` (every history collapses to one belief), and a rank-2 word `f` (beliefs collapse onto a
  segment). Each token `x` acts on beliefs as the projective map `η ↦ ηT⁽ˣ⁾ / ηT⁽ˣ⁾1`, whose image has dimension
  `rank(T⁽ˣ⁾) − 1`, which gives the strata to look for: a point after `s`, a segment after `f`, a fractal after folds.
- **Model.** 143,846 parameters (4 layers, d_model 64, one 8-dimensional head per layer, MLP 256, pre-LayerNorm),
  context 16, trained on fresh samples: AdamW, lr 1e-3, cosine decay, weight decay 0, batch 128, 40k steps.
- **Exact tools.** Sampling, the Bayes belief, the optimal next-token distribution and KL to it are computed
  analytically (`hmmstrata/exact.py`) and unit-tested.

## Results (3 seeds, 40k steps)

| Metric | Value, seeds 0 / 1 / 2 |
|---|---|
| Excess KL to Bayes (10⁻³ nats) | 0.087 / 0.085 / 0.082 |
| Linear belief probe, held-out R² | 0.9965 / 0.9963 / 0.9958 |
| Point-stratum variance ratio (`s` vs fold token 0 at k ≥ 5, centred per position, positions 6–14) | 0.057 / 0.030 / 0.059 |

These reproduce the numbers of an earlier private write-up of the same setup (point ratio 0.056, excess KL about
0.08×10⁻³) once its analysis is corrected for an untrained position. Emitting fold tokens from the *source* state
instead of the destination state gives a visibly different result (point ratio 0.09 to 0.20).

**Learning order** (log-spaced checkpoints, 8 runs; the grid is coarse and every threshold is a choice):

| Event | First checkpoint reaching it |
|---|---|
| KL < 5×10⁻³ nats (coarse surface) | step 400 – 1,600 |
| Belief-probe R² > 0.99 | step 4,000 (every run) |
| KL < 1×10⁻³ nats (fine surface) | step 10,000 – 16,000 |
| Surplus pre-sync history decodable at k = 2, 90% of its gain | step 4,000 – 25,000 |

The coarse next-token surface is learned before the belief geometry; the last order of magnitude of KL arrives after
the belief probe has saturated, together with the surplus history. Measured as "90% of the total KL improvement on
a linear scale", KL looks finished by step 16–40, which is an artefact of its large starting value.

**Surplus history.** After `s` the true belief is a single point, so the token before `s` carries no information about
the future. It is nevertheless decodable from the residual stream: about 0.9 one step after `s`, falling to chance by
about 8 steps (majority baseline 0.23).

**Error inheritance** (does the model build its belief at t+1 from its belief at t?). Real minus shuffled-pair
correlation of probe error at t+1 with probe error at t pushed through the true update. About +0.4 to +0.5 at
initialisation, decaying slowly to roughly 0 to +0.15 at depths 3–6 on the final models. The final models compute each
position mostly afresh, with a weak residual inheritance on some seeds. Depth 1 is negative on most seeds and unexplained.

**What did not work.** TwoNN and correlation-dimension estimates on the residual stream did not discriminate the
strata. Calibration on synthetic sets of known dimension with matched noise shows why: TwoNN reads 7 to 9 whatever
the true dimension. The true belief geometry, computed from the HMM, is the reliable reference.

## Layout

```
hmmstrata/
  process/       base.py (token-labelled HMM), registry.py, sync_strata.py (the process above)
  exact.py       sampling, beliefs, Bayes predictor, KL, distance to events
  model.py       train.py      small GPT and training with log-spaced checkpoints
  analyze.py     final-checkpoint metrics (KL, belief probe, point ratio)
  probes.py      belief and pre-sync-history probes across layers
  series.py      metrics across checkpoints
  inherit.py     error inheritance
  localdim.py  split.py  calibrate.py     the dimension estimators and their calibration
tests/           unit tests (python -m unittest discover -s tests)
notebooks/       hmm_strata_baseline.ipynb (executed), build_notebook.py, results.json, precompute.py
```

Run locally: `python -m hmmstrata.train --process sync_strata --pparams '{"dest_emission": true}' --wd 0 --seed 0 --steps 40000 --device cuda --out runs/s0`,
then `python -m hmmstrata.analyze runs/s0`. The notebook is rebuilt with `python notebooks/build_notebook.py`
(`precompute.py` needs the local `runs/` directory and is kept as a record of how `results.json` was produced).

## Adding a process

A process is a token-labelled HMM: `T[x, i, j] = P(emit x and move to state j | state i)` with `sum_x T[x]`
row-stochastic, plus optional annotations (state groups, token and state events, an `expected` dictionary of
predictions written down before training). Register it with `@register("name")` and the sampler, exact tools,
training and metrics work unchanged; the notebook has a runnable example (an asymmetric variant). Metrics that
look up the tokens `s` and `f` assume the process keeps them.

## Limits

One HMM, one architecture and context length, three seeds in the shipped results. Seeds disagree noticeably on the
point ratio and on how much history is retained. Probe-based measurements are only as good as the linear probe. The
point-stratum variance ratio measures spread, not dimension.

## References

- Shai, Marzen, Teixeira, Gietelink Oldenziel, Riechers (2024). *Transformers represent belief state geometry in their residual stream.* <https://arxiv.org/abs/2405.15943>
- Piotrowski, Riechers, Filan, Shai (ICML 2025). *Constrained belief updates explain geometric structures in transformer representations.* <https://arxiv.org/abs/2502.01954>
- Robinson, Dey, Chiang (NeurIPS 2025). *Token embeddings violate the manifold hypothesis.* <https://arxiv.org/abs/2504.01002>
- Robinson, Dey, Chiang (2024). *The structure of the token space for large language models.* <https://arxiv.org/abs/2410.08993>
- Wang et al. (NeurIPS 2025). *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning.* <https://arxiv.org/abs/2506.01939>

The setup follows an earlier private write-up produced with a Claude model. The checks here (the position artefact in
its analysis, the emission convention, learning order, inheritance, the failed dimension estimators) were done
independently.
