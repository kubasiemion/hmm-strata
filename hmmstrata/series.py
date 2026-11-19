"""Metrics across all checkpoints of a run -> series.json; 90%-of-improvement step per metric."""
import json, sys
from pathlib import Path
import numpy as np
from .probes import evaluate


def series(run: Path, n=12000, dev="xpu"):
    out = []
    for p in sorted(run.glob("ckpt_*.pt")):
        s = int(p.stem.split("_")[1])
        r = evaluate(run, s, n=n, dev=dev, ks=(2, 4), layers_k=2)
        out.append(dict(step=s, kl_e3=r["kl_e3"], belief_r2=r["belief_r2"][-1],
                        hist_k2=r["presync_final"][2][0], hist_k4=r["presync_final"][4][0]))
    (run / "series.json").write_text(json.dumps(out, indent=1))
    return out


def t90(steps, vals, frac=0.9):
    v = np.array(vals, float); v0, v1 = v[0], v[-1]
    prog = (v - v0) / (v1 - v0)
    return int(steps[int(np.argmax(prog >= frac))])


def summary(out):
    st = [o["step"] for o in out]
    return {k: t90(st, [o[k] for o in out]) for k in ("kl_e3", "belief_r2", "hist_k2")}


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        run = Path(f"runs/control_sync_strata/s{s}")
        out = series(run)
        print(f"seed {s}: 90% step {summary(out)}  hist_k2 {out[0]['hist_k2']:.2f}->{out[-1]['hist_k2']:.2f}  hist_k4 {out[0]['hist_k4']:.2f}->{out[-1]['hist_k4']:.2f}", flush=True)
