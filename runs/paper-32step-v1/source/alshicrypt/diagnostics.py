from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .data import load_split, make_manifest, save_json, seed_for
from .evaluate import metrics, summarize
from . import processes as p


def diagnose(root: Path, output: Path):
    torch.set_num_threads(4)
    manifest = make_manifest(root, output / "split.json")
    train, _ = load_split(manifest, root, "train")
    val, ids = load_split(manifest, root, "val")
    # Retrieval by silhouette alone, without access to RGB at prediction time.
    a = (train[:, 3].float() / 255).flatten(1)
    b = (val[:, 3].float() / 255).flatten(1)
    distances = torch.cdist(b, a)
    retrieved = train[distances.argmin(1)].float()
    retrieved[:, 3] = val[:, 3]
    alpha_only = summarize(metrics(retrieved, val.float(), val[:, 3:], 255))
    corrupted = p.old_corruption(val.float())
    distinct = {hashlib.sha256(row[:3].numpy().tobytes()).hexdigest() for row in corrupted}
    g = torch.Generator().manual_seed(seed_for("gaussian-quantization-diagnostic", "val"))
    x = val.double() / 255
    n = p.noise_field(tuple(x.shape), "gaussian", g)
    y = p.forward(x, n, "gaussian")
    quantized = (y.clamp(0, 1) * 255).round() / 255
    recovered = p.inverse(quantized, n, "gaussian")
    gaussian = summarize(metrics(recovered, x, val[:, 3:], 1))
    reference = p.inverse(y, n, "gaussian")
    result = {"split": "val", "ids": ids, "old_process": {"unique_rgb_arrays": len(distinct), "images": len(val), "alpha_unchanged": bool(torch.equal(corrupted[:, 3], val[:, 3].float())), "saturation_fraction": float(((corrupted[:, :3] == 0) | (corrupted[:, :3] == 255)).float().mean())}, "alpha_only_nearest_training_mask": alpha_only, "gaussian_png_clipping_and_quantization": gaussian, "gaussian_float64_max_roundtrip_error": float((reference - x).abs().max())}
    save_json(output / "diagnostics.json", result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(8, 8), constrained_layout=True)
    for i in range(4):
        axes[i, 0].imshow(val[i].permute(1, 2, 0).numpy())
        axes[i, 1].imshow(corrupted[i].to(torch.uint8).permute(1, 2, 0).numpy())
        axes[i, 2].imshow(corrupted[i, :3].to(torch.uint8).permute(1, 2, 0).numpy())
        axes[i, 3].imshow(val[i, 3].numpy(), cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_ylabel(ids[i])
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, title in zip(axes[0], ["Original", "Old stored RGBA", "Old RGB only", "Unchanged alpha"]):
        ax.set_title(title)
    fig.savefig(output / "old-process.pdf")
    fig.savefig(output / "old-process.png", dpi=160)
    plt.close(fig)
    # Cross the training/evaluation noise modes. Comparing only matched modes
    # cannot distinguish learning the family from fitting one realized field.
    cross = {}
    if (output / "promotion.json").exists():
        from .campaign import run_name, config
        from .evaluate import evaluate_domains
        from .models import make_model
        from .train import setup
        families = json.loads((output / "promotion.json").read_text())["families"]
        for track, family in families.items():
            for training_noise, phase in [("fresh", "extended"), ("fixed", "diagnostics")]:
                name = run_name(config(track, family, "mixed", noise_mode=training_noise))
                checkpoint = output / phase / name / "best.pt"
                if not checkpoint.exists():
                    continue
                device = setup(17, require_cuda=False)
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model = make_model(family, track, saved["config"]["width"]).to(device)
                model.load_state_dict(saved["model"])
                for evaluation_noise in ["fresh", "fixed"]:
                    cross[f"{track}_{training_noise}_{evaluation_noise}"] = {
                        "track": track, "training_noise": training_noise,
                        "evaluation_noise": evaluation_noise, "split": "val",
                        "results": evaluate_domains(model, val, ids, track, device, "val", evaluation_noise)}
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    if cross:
        save_json(output / "noise-cross-validation.json", cross)
    print(json.dumps(result, indent=2), flush=True)
    return result
