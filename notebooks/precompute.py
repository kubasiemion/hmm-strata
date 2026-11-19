"""Collect the precomputed results the notebook ships with -> notebooks/results.json."""
import json
from pathlib import Path
import numpy as np
from hmmstrata.analyze import analyze
from hmmstrata.inherit import inheritance
R = Path("runs"); out = {"final": {}, "series": {}, "inherit_dev": {}, "inherit_final": {}}
for tag, ns in (("dest", 3), ("control", 5)):
    for s in range(ns):
        run = R / f"{tag}_sync_strata/s{s}"
        out["series"][f"{tag}{s}"] = json.loads((run / "series.json").read_text())
for s in range(3):
    run = R / f"dest_sync_strata/s{s}"
    a = analyze(run, dev="cpu"); out["final"][f"dest{s}"] = {k: v for k, v in a.items() if k != "ckpt"}
    out["inherit_final"][f"dest{s}"] = {k: v[:2] for k, v in inheritance(run, n=30000, dev="cpu").items()}
    for st in (0, 10, 40, 100, 399, 1003, 2521, 6334, 15918, 25233, 40000):
        r = inheritance(run, st, n=30000, dev="cpu", depths=(3, 5))
        out["inherit_dev"].setdefault(f"dest{s}", {})[st] = {k: v[:2] for k, v in r.items()}
    print("done", s, flush=True)
Path("notebooks/results.json").write_text(json.dumps(out, default=float))
