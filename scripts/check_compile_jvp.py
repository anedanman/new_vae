"""Check that torch.compile gives the same pMF outputs and JVP as eager."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import mit  # noqa: E402
from models.pmf import pMF_models  # noqa: E402

torch.manual_seed(0)
m = pMF_models["pMF_B"](img_size=128, patch_size=16).cuda()
# give the zero-initialized gates / output layers some weight so outputs are non-trivial
with torch.no_grad():
    for n, p in m.named_parameters():
        if p.abs().sum() == 0:
            p.normal_(0, 0.05)
B = 8
z = torch.randn(B, 3, 128, 128, device="cuda")
t = torch.rand(B, 1, 1, 1, device="cuda")
r = t * torch.rand(B, 1, 1, 1, device="cuda")
omega = torch.ones(B, 1, 1, 1, device="cuda") * 2
tmin = torch.zeros_like(t); tmax = torch.ones_like(t)
y = torch.randint(0, 1000, (B,), device="cuda")
tan = torch.randn_like(z)


def run():
    f = lambda zz, tt, rr: m.u_fn(zz, tt, tt - rr, omega, tmin, tmax, y)
    mit.ATTN_IMPL["value"] = "einsum"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        u, du, v = torch.func.jvp(f, (z, t, r), (tan, torch.ones_like(t), torch.zeros_like(r)), has_aux=True)
    mit.ATTN_IMPL["value"] = "sdpa"
    return u.float(), du.float(), v.float()


eager = run()
m.net.compile()
comp = run()
for name, a, b in zip(("u", "du_dt", "v"), eager, comp):
    rel = (a - b).norm() / a.norm()
    print(f"{name}: rel diff eager vs compiled = {rel.item():.2e}")
