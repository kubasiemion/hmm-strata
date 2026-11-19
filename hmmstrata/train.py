"""Train a GPT on fresh samples of a registered process; log-spaced checkpoints."""
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from .process import make
from . import exact
from .model import GPT


def log_steps(total: int, n: int = 24) -> list[int]:
    s = {0, total} | {int(round(x)) for x in np.geomspace(1, total, n)}
    return sorted(s)


def device(name: str):
    return torch.device(name)


def train(process: str, pparams: dict, seed: int, steps: int, out: Path, dev: str = "cpu",
          ctx: int = 16, batch: int = 128, lr: float = 1e-3, model_kw: dict | None = None,
          n_ckpt: int = 24, wd: float = 0.01):
    hmm = make(process, **pparams)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GPT(hmm.V, ctx=ctx, **(model_kw or {})).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(
        process=process, pparams=pparams, seed=seed, steps=steps, ctx=ctx, batch=batch, lr=lr,
        model=model.cfg, n_params=nparams, device=dev, wd=wd), indent=1))
    ckpts = set(log_steps(steps, n_ckpt))
    log = open(out / "train_log.jsonl", "w")
    t0 = time.time()

    def save(step):
        torch.save(model.state_dict(), out / f"ckpt_{step:06d}.pt")

    for step in range(steps + 1):
        if step in ckpts:
            save(step)
        if step == steps:
            break
        toks, _ = exact.sample(hmm, batch, ctx + 1, rng)
        x = torch.from_numpy(toks).to(dev)
        loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, hmm.V), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 500 == 0:
            log.write(json.dumps(dict(step=step, loss=loss.item(), t=time.time() - t0)) + "\n"); log.flush()
    return nparams


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="sync_strata")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--pparams", default="{}")
    a = ap.parse_args()
    print("params", train(a.process, json.loads(a.pparams), a.seed, a.steps, Path(a.out), a.device, wd=a.wd))
