from __future__ import annotations

import time

import numpy as np
import torch

from . import processes as p
from .data import probes, seed_for, synthetic_set


def metrics(pred: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor, scale: float) -> dict:
    diff = (pred.double() - target.double()) / scale
    absolute = diff.abs()
    mse = diff.square().flatten(1).mean(1)
    exact = pred.round().eq(target.round()) if scale == 255 else (absolute <= 1e-6)
    fg = (alpha > 0).expand(-1, 3, -1, -1)
    foreground = (absolute[:, :3] * fg).flatten(1).sum(1) / fg.flatten(1).sum(1).clamp_min(1)
    return {"mae": absolute.flatten(1).mean(1).tolist(), "mse": mse.tolist(), "max_error": absolute.flatten(1).amax(1).tolist(), "byte_accuracy" if scale == 255 else "within_1e-6": exact.double().flatten(1).mean(1).tolist(), "exact_image_rate" if scale == 255 else "images_within_1e-6": exact.flatten(1).all(1).double().tolist(), "foreground_rgb_mae": foreground.tolist(), "rgb_mae": absolute[:, :3].flatten(1).mean(1).tolist()}


def summarize(records: dict) -> dict:
    result = {key: float(np.mean(values)) for key, values in records.items()}
    result["max_error"] = max(records["max_error"])
    result["psnr"] = float(-10 * np.log10(result["mse"])) if result["mse"] > 0 else None
    return result


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, images: torch.Tensor, ids: list[str], track: str, device: torch.device, split: str, domain: str, noise_mode: str = "fresh", batch: int = 4, per_image: bool = False) -> dict:
    model.eval()
    g = torch.Generator().manual_seed(seed_for("evaluation-noise", split, domain))
    fixed_g = torch.Generator().manual_seed(seed_for("fixed-noise", 17))
    fixed_n = p.noise_field((1, *images.shape[1:]), track, fixed_g)
    all_records = {k: {} for k in ["forward", "reverse", "roundtrip", "analytic"]}
    payload = 0
    inference_seconds = 0.
    for start in range(0, len(images), batch):
        raw = images[start:start + batch].float()
        ref_x = raw.double() if track == "byte" else raw.double() / 255
        n64 = fixed_n.expand_as(ref_x) if noise_mode == "fixed" else p.noise_field(tuple(raw.shape), track, g)
        y64 = p.forward(ref_x, n64, track)
        x, n, y = ref_x.float().to(device), n64.float().to(device), y64.float().to(device)
        synchronize(device)
        t = time.perf_counter()
        encoded = model.encode(x, n)
        reverse = model.decode(y, n)
        synchronize(device)
        inference_seconds += time.perf_counter() - t
        transmitted, size = p.serialize_roundtrip(encoded, track)
        payload += size
        synchronize(device)
        t = time.perf_counter()
        recovered = model.decode(transmitted, n)
        synchronize(device)
        inference_seconds += time.perf_counter() - t
        actual_y, _ = p.serialize_roundtrip(y, track)
        analytic = p.inverse(actual_y, n, track)
        outputs = {"forward": (encoded, y), "reverse": (reverse, x), "roundtrip": (recovered, x), "analytic": (analytic, x)}
        for name, (pred, target) in outputs.items():
            rec = metrics(pred, target, raw[:, 3:].to(device), 255 if track == "byte" else 1)
            for key, values in rec.items():
                all_records[name].setdefault(key, []).extend(values)
    result = {key: summarize(value) for key, value in all_records.items()}
    result.update({"count": len(images), "payload_bytes_per_image": payload / len(images), "noise_field_bytes_per_image": images[0].numel() * (1 if track == "byte" else 4), "seed_bytes_alternative": 8, "neural_seconds_per_image_three_calls": inference_seconds / len(images)})
    if per_image:
        result["per_image"] = {"ids": ids, "metrics": all_records}
    return result


def evaluate_domains(model, pokemon, ids, track, device, split, noise_mode="fresh", per_image=False):
    static = synthetic_set(split)
    probe_images, probe_ids = probes()
    return {"pokemon": evaluate(model, pokemon, ids, track, device, split, "pokemon", noise_mode, per_image=per_image), "static": evaluate(model, static, [f"static_{i}" for i in range(len(static))], track, device, split, "static", noise_mode, per_image=per_image), "probes": evaluate(model, probe_images, probe_ids, track, device, split, "probes", noise_mode, per_image=per_image)}


def selection_score(result: dict, track: str) -> tuple:
    m = result["pokemon"]
    if track == "byte":
        return (-m["roundtrip"]["exact_image_rate"], -m["forward"]["byte_accuracy"], -m["reverse"]["byte_accuracy"], m["roundtrip"]["mae"])
    # Treat differences below float transport tolerance as equivalent, rather
    # than selecting an unlearned map for a meaningless rounding advantage.
    return (max(0., m["roundtrip"]["mae"] - 1e-6), m["forward"]["mae"] + m["reverse"]["mae"])
