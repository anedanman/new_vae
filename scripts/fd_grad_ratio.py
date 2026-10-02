"""Gradient norms of the VAE loss vs the FD-loss branch at a checkpoint (to set --fd_weight).

python scripts/fd_grad_ratio.py <ckpt> [micro] [batches]
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fd_loss import InceptionFDLoss  # noqa: E402
from models.perceptual_loss import PerceptualLoss  # noqa: E402
from train import build_model, fd_seed_moments, fd_step  # noqa: E402
from utils.data import ImageNet128, to_model_input  # noqa: E402
from utils.misc import setup_logging  # noqa: E402

setup_logging()
ckpt = sys.argv[1]
micro = int(sys.argv[2]) if len(sys.argv) > 2 else 32
n_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 3
state = torch.load(os.path.expanduser(ckpt), map_location="cuda", weights_only=False)
a = SimpleNamespace(**state["args"])
a.micro_batch, a.fd_batch, a.fd_weight, a.fd_init_samples, a.gen_bsz = micro, 256, 1.0, 20000, 250
model = build_model(a).cuda().train()
model.load_state_dict(state["model"])
del state
perc = PerceptualLoss(lpips_weight=0.4, convnext_weight=0.1, random_crop=True).cuda()
fd = InceptionFDLoss(os.path.join(os.path.expanduser(a.ref_dir), "inception_train.npz")).cuda()
fd_seed_moments(model, fd, a)
train = ImageNet128(os.path.expanduser(a.data_root), "train")
rng = np.random.default_rng(0)


def gnorm():
    return torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None)).item()


for i in range(n_batches):
    imgs, y = train.get(rng.choice(len(train), 256, replace=False))
    x, y = to_model_input(torch.from_numpy(imgs), "cuda"), torch.from_numpy(y).cuda()
    model.zero_grad(set_to_none=True)
    for k in range(256 // micro):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(x[k * micro:(k + 1) * micro], y[k * micro:(k + 1) * micro], aux_loss_fn=perc,
                            update_stats=False)
        (loss / (256 // micro)).backward()
    g_vae = gnorm()
    g_vae_vec = torch.cat([p.grad.flatten().float() for p in model.decoder.parameters() if p.grad is not None])
    model.zero_grad(set_to_none=True)
    a.fd_weight_mode, a.fd_ramp_steps, a.fd_start_step = "fixed", 0, 0
    fd_val, _ = fd_step(model, fd, a, micro, 0)
    g_fd = gnorm()
    g_fd_vec = torch.cat([p.grad.flatten().float() for p in model.decoder.parameters() if p.grad is not None])
    cos = torch.nn.functional.cosine_similarity(g_vae_vec, g_fd_vec, dim=0).item()
    print(f"batch {i}: FD={fd_val.item():.2f}  |g_vae|={g_vae:.3f}  |g_fd(w=1)|={g_fd:.3f}  "
          f"ratio={g_fd / g_vae:.1f}  cos(decoder grads)={cos:.3f}")
