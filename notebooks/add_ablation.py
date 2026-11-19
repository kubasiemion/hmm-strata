"""Add the history-ablation results (3 destination-emission seeds) to notebooks/results.json."""
import json
from pathlib import Path
from hmmstrata.ablate import run
R = json.loads(Path("notebooks/results.json").read_text())
R["ablation"] = {}
for s in range(3):
    rows = run(Path(f"runs/dest_sync_strata/s{s}"), dev="xpu")
    R["ablation"][f"dest{s}"] = {str(k): v for k, v in rows.items()}
    print("done", s, flush=True)
Path("notebooks/results.json").write_text(json.dumps(R, default=float))
