"""Evaluate a saved milestone offline (same protocol as the in-training eval).

    python eval_ckpt.py --ckpt ~/new_vae/runs/<exp>/checkpoints/milestone_0050000.pt [--wandb_id <id>]

Results go to <run_dir>/eval_<step>.json and, with --wandb_id, into that wandb run
(eval metrics use the "eval_step" x-axis, so late/offline evals line up with live ones).
"""
import argparse
import json
import os
from types import SimpleNamespace

import torch

from metrics.evaluator import Evaluator
from train import build_model, run_eval
from utils.data import ImageNet128
from utils.ema import EMAModel
from utils.misc import setup_logging

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ref_dir", default=None)
    p.add_argument("--eval_num", type=int, default=None)
    p.add_argument("--sweep_num", type=int, default=None)
    p.add_argument("--cfg_grid", type=float, nargs="+", default=None)
    p.add_argument("--wandb_id", default=None)
    cli = p.parse_args()
    setup_logging()

    state = torch.load(os.path.expanduser(cli.ckpt), map_location="cuda", weights_only=False)
    args = SimpleNamespace(**state["args"])
    for k in ("ref_dir", "eval_num", "sweep_num", "cfg_grid"):
        if getattr(cli, k) is not None:
            setattr(args, k, getattr(cli, k))
    step = state["step"]

    model = build_model(args).cuda()
    model.load_state_dict(state["model"])
    ema = EMAModel(model, ema_type="edm", values=args.ema_halflife_kimg, batch_size=args.batch_size)
    ema.load_state_dict(state["ema"])
    evaluator = Evaluator(args.ref_dir)
    assert evaluator.usable, f"no Inception reference stats in {args.ref_dir}"
    val = ImageNet128(args.data_root, "val")

    logs = run_eval(args, model, ema, evaluator, step, val)
    logs["eval_step"] = step
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.expanduser(cli.ckpt))), f"eval_{step:07d}.json")
    json.dump(logs, open(out, "w"), indent=2)
    print(f"wrote {out}")

    if cli.wandb_id:
        import wandb
        run = wandb.init(project=args.wandb_project, id=cli.wandb_id, resume="must")
        run.define_metric("eval_step")
        for pattern in ("eval/*", "eval_cfg1/*", "eval_best/*", "eval_recon/*", "sweep/*"):
            run.define_metric(pattern, step_metric="eval_step")
        run.log(logs)
        run.finish()


if __name__ == "__main__":
    main()
