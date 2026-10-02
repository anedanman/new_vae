"""Check that compiling the whole pMF training forward (incl. jvp + perceptual loss) matches eager.

inductor's fallback_random makes compiled code draw the same random numbers as eager,
so losses and parameter gradients are directly comparable.
"""
import os
import sys

import torch
import torch._inductor.config as inductor_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.perceptual_loss import PerceptualLoss  # noqa: E402
from models.pmf import pMF_models  # noqa: E402

inductor_config.fallback_random = True
torch.manual_seed(0)
m = pMF_models["pMF_B"](img_size=128, patch_size=16).cuda()
with torch.no_grad():  # make zero-initialized gates / heads non-trivial
    for n, p in m.named_parameters():
        if p.abs().sum() == 0:
            p.normal_(0, 0.05)
perc = PerceptualLoss(lpips_weight=0.4, convnext_weight=0.1, random_crop=True).cuda()
B = 16
x = torch.rand(B, 3, 128, 128, device="cuda") * 2 - 1
y = torch.randint(0, 1000, (B,), device="cuda")


def run(model, loss_fn, bf16=True):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        loss, ld = model(x, y, aux_loss_fn=loss_fn)
    loss.backward()
    grads = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.grad is not None])
    return loss.detach().float(), grads, {k: float(v) for k, v in ld.items()}


e1 = run(m, perc)
e2 = run(m, perc)
f32 = run(m, perc, bf16=False)
for name, i in (("loss", 0), ("param_grads", 1)):
    print(f"eager-fp32 {name:12s} rel diff vs eager-bf16 = {((e1[i] - f32[i]).norm() / f32[i].norm()).item():.2e}")
m.compile()
perc.compile()
c1 = run(m, perc)
c2 = run(m, perc)  # second call: steady-state compiled graph
for tag, c in (("compiled#1", c1), ("compiled#2", c2)):
    for name, i in (("loss", 0), ("param_grads", 1)):
        rel = ((e1[i] - c[i]).norm() / e1[i].norm()).item()
        base = ((e1[i] - e2[i]).norm() / e1[i].norm()).item()
        rel32 = ((f32[i] - c[i]).norm() / f32[i].norm()).item()
        print(f"{tag} {name:12s} rel diff vs eager = {rel:.2e}   (eager vs eager = {base:.2e}; vs fp32 = {rel32:.2e})")
print("eager   :", {k: round(v, 5) for k, v in e1[2].items()})
print("compiled:", {k: round(v, 5) for k, v in c2[2].items()})
