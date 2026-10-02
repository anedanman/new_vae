"""Visualization panels logged every vis_every steps (fixed classes, noise and images)."""
import io

import numpy as np
import torch
from torchvision.utils import make_grid

# FD-loss's classes of interest + 8 more diverse ones
VIS_CLASSES = [207, 360, 387, 974, 88, 979, 417, 279, 1, 130, 292, 323, 511, 717, 812, 980]


def _grid(x, nrow):
    """x in [-1, 1], (N, 3, H, W) -> uint8 HWC numpy."""
    g = make_grid(((x.float().clamp(-1, 1) + 1) / 2).cpu(), nrow=nrow, padding=2, pad_value=1.0)
    return (g.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _fig_to_array(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return np.array(Image.open(buf).convert("RGB"))


class Visualizer:
    def __init__(self, model_kind, model, val_images, val_labels, cfg_grid, device="cuda",
                 samples_per_class=8, seed=1234, pmf_interval=(0.1, 0.7)):
        self.kind = model_kind
        self.device = device
        self.cfg_grid = cfg_grid
        self.pmf_interval = pmf_interval
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = len(VIS_CLASSES) * samples_per_class
        self.class_labels = torch.tensor(VIS_CLASSES, device=device).repeat_interleave(samples_per_class)
        self.samples_per_class = samples_per_class
        if model_kind == "pmf":
            shape = (model.in_channels, model.input_size, model.input_size)
        else:
            shape = model.latent_shape
        self.noise = torch.randn((n,) + tuple(shape), generator=g).to(device)
        self.sweep_noise = torch.randn((8,) + tuple(shape), generator=g).to(device)
        self.val_x = val_images.to(device)  # (16, 3, H, W) in [-1, 1]
        self.val_y = val_labels.to(device)
        self.torch_gen_seed = seed

    def _gen(self, model, labels, cfg, noise, **kw):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.kind == "pmf":
                return model.generate(labels, cfg=cfg, t_min=self.pmf_interval[0],
                                      t_max=self.pmf_interval[1], noise=noise, **kw)
            return model.generate(labels, cfg=cfg, noise=noise, **kw)

    @torch.no_grad()
    def panels(self, model, vis_cfg):
        """Returns {name: uint8 HWC image} and {name: scalar}."""
        imgs, scalars = {}, {}
        spc = self.samples_per_class
        for cfg in sorted({1.0, vis_cfg}):
            x = self._gen(model, self.class_labels, cfg, self.noise)
            imgs[f"samples/class_grid_cfg{cfg:g}"] = _grid(x, nrow=spc)

        # cfg sweep: rows = classes, columns = cfg values, same noise per row
        labels = torch.tensor(VIS_CLASSES[:8], device=self.device)
        cols = [self._gen(model, labels, c, self.sweep_noise) for c in self.cfg_grid]
        x = torch.stack(cols, dim=1).flatten(0, 1)
        imgs["samples/cfg_sweep"] = _grid(x, nrow=len(self.cfg_grid))

        if self.kind == "pmf":
            imgs.update(self._pmf_panels(model, vis_cfg))
        elif self.kind == "hvae":
            imgs.update(self._hvae_panels(model, vis_cfg))
        else:
            p, s = self._vae_panels(model, vis_cfg)
            imgs.update(p)
            scalars.update(s)
        return imgs, scalars

    def _pmf_panels(self, model, vis_cfg):
        out = {}
        # few-step vs one-step, same noise
        labels = torch.tensor(VIS_CLASSES[:8], device=self.device)
        rows = [self._gen(model, labels, vis_cfg, self.sweep_noise, num_steps=k) for k in (1, 2, 4)]
        out["pmf/steps_1_2_4"] = _grid(torch.cat(rows), nrow=8)
        # x-prediction from noised val images
        torch.manual_seed(self.torch_gen_seed)
        x, y = self.val_x[:8], self.val_y[:8]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res = model.denoise_preview(x, y, ts=(0.2, 0.5, 0.8, 1.0))
        cols = [x]
        for z_t, x_hat in res:
            cols += [z_t / max(1.0, float(model.noise_scale)), x_hat]
        out["pmf/denoise_t0.2_0.5_0.8_1.0"] = _grid(torch.stack(cols, 1).flatten(0, 1), nrow=len(cols))
        return out

    def _vae_panels(self, model, vis_cfg):
        out, scal = {}, {}
        x, y = self.val_x, self.val_y
        torch.manual_seed(self.torch_gen_seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, logvar = model.encode(x, y)
            rec_mean = model.decode(mu, y)
            rec_samp = model.decode(mu + torch.exp(0.5 * logvar) * torch.randn_like(mu), y)
        out["vae/recon_orig_mean_sampled"] = _grid(torch.cat([x, rec_mean, rec_samp]), nrow=x.shape[0])

        # prior temperature sweep
        labels = torch.tensor(VIS_CLASSES[:8], device=self.device)
        temps = (0.0, 0.5, 0.75, 1.0, 1.25)
        cols = [self._gen(model, labels, vis_cfg, self.sweep_noise, temperature=t) for t in temps]
        out["vae/temperature_0_.5_.75_1_1.25"] = _grid(torch.stack(cols, 1).flatten(0, 1), nrow=len(temps))

        # interpolation between posterior means of val pairs
        steps = torch.linspace(0, 1, 8, device=self.device)
        rows = []
        for a, b in ((0, 1), (2, 3), (4, 5), (6, 7)):
            z = (1 - steps)[:, None, None] * mu[a] + steps[:, None, None] * mu[b]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rows.append(model.decode(z, y[a].expand(8)))
        out["vae/interp"] = _grid(torch.cat(rows), nrow=8)

        # class swap: same posterior mean, other labels (how much the decoder uses y)
        swap_labels = VIS_CLASSES[8:12]
        rows = [x[:8], rec_mean[:8]]
        for c in swap_labels:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rows.append(model.decode(mu[:8], torch.full((8,), c, device=self.device)))
        out["vae/class_swap"] = _grid(torch.stack(rows, 1).flatten(0, 1), nrow=len(rows))

        R = getattr(model, "num_registers", 0)
        if R > 0:
            # rows: one register draw each; columns: independent patch-token draws (same class)
            g = torch.Generator(device="cpu").manual_seed(self.torch_gen_seed + 1)
            regs = torch.randn((6, R, model.latent_dim), generator=g).to(self.device)
            patches = torch.randn((6, model.num_patches, model.latent_dim), generator=g).to(self.device)
            z = torch.cat([regs[:, None].expand(-1, 6, -1, -1), patches[None].expand(6, -1, -1, -1)], dim=2)
            labels = torch.full((36,), VIS_CLASSES[0], device=self.device)
            x_rc = self._gen(model, labels, vis_cfg, z.flatten(0, 1))
            out["vae/register_rows_patch_cols"] = _grid(x_rc, nrow=6)
        return out, scal

    def _hvae_panels(self, model, vis_cfg):
        out = {}
        x, y = self.val_x, self.val_y
        torch.manual_seed(self.torch_gen_seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec_mean = model.reconstruct(x, y, sample=False)
            rec_samp = model.reconstruct(x, y, sample=True)
        out["vae/recon_orig_mean_sampled"] = _grid(torch.cat([x, rec_mean, rec_samp]), nrow=x.shape[0])

        # rows: Euler steps per stage (0 = plain HVAE prior samples), same noise
        labels = torch.tensor(VIS_CLASSES[:8], device=self.device)
        steps = (0, 1, 4, 16)
        rows = [self._gen(model, labels, vis_cfg, self.sweep_noise, flow_steps=k) for k in steps]
        out["hvae/flow_steps_" + "_".join(map(str, steps))] = _grid(torch.cat(rows), nrow=8)

        temps = (0.0, 0.5, 0.75, 1.0, 1.25)
        cols = [self._gen(model, labels, vis_cfg, self.sweep_noise, temperature=t) for t in temps]
        out["vae/temperature_0_.5_.75_1_1.25"] = _grid(torch.stack(cols, 1).flatten(0, 1), nrow=len(temps))

        # what each stage encodes: columns = original, then posterior for the first k stages and
        # the generative path (cfg 1) for the rest; same draws in every column
        L = model.num_stages
        ks = sorted({0, 1, 2, L // 2, L - 1, L})
        cols = [x[:8]]
        for k in ks:
            torch.manual_seed(self.torch_gen_seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cols.append(model.reconstruct(x[:8], y[:8], sample=True, post_stages=k))
        out["hvae/posterior_stages_" + "_".join(map(str, ks))] = _grid(
            torch.stack(cols, 1).flatten(0, 1), nrow=len(cols))
        return out


@torch.no_grad()
def latent_diagnostics(model, x, y, bsz=256):
    """Statistics of the aggregate posterior over a set of val images (VAEs only).

    Returns (scalars, {name: uint8 plot image}).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mus, stds, zs = [], [], []
    for lo in range(0, x.shape[0], bsz):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, logvar = model.encode(x[lo: lo + bsz], y[lo: lo + bsz])
        std = torch.exp(0.5 * logvar)
        mus.append(mu.float())
        stds.append(std.float())
        zs.append((mu + std * torch.randn_like(mu)).float())
    mu, std, z = torch.cat(mus), torch.cat(stds), torch.cat(zs)  # (N, R + T, C)
    R = getattr(model, "num_registers", 0)
    reg_mu, reg_std = mu[:, :R], std[:, :R]
    mu, std, z = mu[:, R:], std[:, R:], z[:, R:]  # spatial statistics use patch tokens only
    N, T, C = z.shape
    side = int(T ** 0.5)

    zf = z.flatten(1)
    var_dim = zf.var(0)
    mean_dim = zf.mean(0)
    # channel covariance pooled over tokens (after removing per-(token, channel) means)
    zc = (z - z.mean(0, keepdim=True)).reshape(-1, C).double()
    cov = zc.T @ zc / (zc.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).flip(0).clamp_min(1e-12).float().cpu().numpy()
    d = cov.diagonal().sqrt()
    corr = cov / (d[:, None] * d[None, :] + 1e-12)
    offdiag = corr[~torch.eye(C, dtype=torch.bool, device=corr.device)]
    # neighbour-token correlation per channel, averaged
    zg = (z - z.mean(0, keepdim=True)).reshape(N, side, side, C)
    a, b = zg[:, :, :-1].reshape(-1, C), zg[:, :, 1:].reshape(-1, C)
    nb_corr = ((a * b).mean(0) / (a.std(0) * b.std(0) + 1e-8)).mean()
    # same on posterior means over informative channels only (inactive dims are pure noise in z)
    mg = (mu - mu.mean(0, keepdim=True)).reshape(N, side, side, C)
    active = mu.reshape(-1, C).var(0) > 0.01
    ma, mb = mg[:, :, :-1].reshape(-1, C)[:, active], mg[:, :, 1:].reshape(-1, C)[:, active]
    nb_corr_mu = ((ma * mb).mean(0) / (ma.std(0) * mb.std(0) + 1e-8)).mean() if active.any() else torch.tensor(0.0)

    scalars = {
        "latent/agg_var_mean": var_dim.mean().item(),
        "latent/agg_var_p05": var_dim.quantile(0.05).item(),
        "latent/agg_var_p95": var_dim.quantile(0.95).item(),
        "latent/agg_mean_abs": mean_dim.abs().mean().item(),
        "latent/post_std_mean": std.mean().item(),
        "latent/post_std_p05": std.flatten()[:: max(1, std.numel() // 1_000_000)].quantile(0.05).item(),
        "latent/chan_eig_top1_frac": float(eig[0] / eig.sum()),
        "latent/chan_eff_rank": float(np.exp(-(eig / eig.sum() * np.log(eig / eig.sum())).sum())),
        "latent/chan_offdiag_corr_rms": offdiag.pow(2).mean().sqrt().item(),
        "latent/neighbor_token_corr": nb_corr.item(),
        "latent/neighbor_token_corr_mu_active": nb_corr_mu.item(),
        "latent/active_channels_frac": active.float().mean().item(),
    }
    if R > 0:
        reg_kl = 0.5 * (reg_mu ** 2 + reg_std ** 2 - 1 - 2 * reg_std.clamp_min(1e-12).log()).mean(0)  # (R, C)
        patch_kl = 0.5 * (mu ** 2 + std ** 2 - 1 - 2 * std.clamp_min(1e-12).log()).mean(0)  # (T, C)
        scalars.update({
            "latent/register_kl_per_token": reg_kl.sum(1).mean().item(),
            "latent/patch_kl_per_token": patch_kl.sum(1).mean().item(),
            "latent/register_active_channels": (reg_kl > 0.01).sum(1).float().mean().item(),
            "latent/patch_active_channels": (patch_kl > 0.01).sum(1).float().mean().item(),
        })

    def _bins(v):  # a near-zero-range histogram (e.g. right after init) cannot have 100 bins
        return 100 if np.ptp(v) > 1e-3 * max(1.0, float(np.abs(v).max())) else 1

    var_np = var_dim.double().cpu().numpy()
    std_np = std.flatten()[:: max(1, std.numel() // 2_000_000)].double().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    axes[0].hist(var_np, bins=_bins(var_np), log=True)
    axes[0].axvline(1.0, color="k", ls="--")
    axes[0].set_title("aggregate var per latent dim (sampled z)")
    axes[1].hist(std_np, bins=_bins(std_np), log=True)
    axes[1].set_title("posterior std")
    axes[2].semilogy(eig)
    axes[2].axhline(1.0, color="k", ls="--")
    axes[2].set_title("channel covariance eigenvalues (prior = 1)")
    return scalars, {"latent/stats": _fig_to_array(fig)}
