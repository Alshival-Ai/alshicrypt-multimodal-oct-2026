from __future__ import annotations

import io
import math

import numpy as np
import torch
from PIL import Image

STEPS = 16
SIGNAL = 0.5
BETA = 1.0 - SIGNAL ** (2 / STEPS)


def noise_field(shape: tuple, track: str, generator: torch.Generator, fixed: bool = False) -> torch.Tensor:
    shape = (1, *shape[1:]) if fixed else shape
    if track == "byte":
        return torch.randint(256, shape, generator=generator).float()
    if track != "gaussian":
        raise ValueError(track)
    # The exact endpoint law of the 16-step Gaussian process.
    return torch.randn(shape, generator=generator, dtype=torch.float64) * math.sqrt(1 - SIGNAL**2)


def forward(x: torch.Tensor, noise: torch.Tensor, track: str) -> torch.Tensor:
    return torch.remainder(x + noise, 256) if track == "byte" else SIGNAL * x + noise


def inverse(y: torch.Tensor, noise: torch.Tensor, track: str) -> torch.Tensor:
    return torch.remainder(y - noise, 256) if track == "byte" else (y - noise) / SIGNAL


def serialize_roundtrip(x: torch.Tensor, track: str) -> tuple[torch.Tensor, int]:
    """Actual lossless PNG or NumPy float32 transport; no hidden image bypass."""
    items, total = [], 0
    for item in x.detach().cpu():
        stream = io.BytesIO()
        if track == "byte":
            array = item.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
            Image.fromarray(array).save(stream, format="PNG")
            total += stream.tell()
            stream.seek(0)
            with Image.open(stream) as im:
                items.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float())
        else:
            np.save(stream, item.float().numpy(), allow_pickle=False)
            total += stream.tell()
            stream.seek(0)
            items.append(torch.from_numpy(np.load(stream, allow_pickle=False)))
    return torch.stack(items).to(x.device), total


def old_corruption(x: torch.Tensor, seed: int = 17) -> torch.Tensor:
    """Original float32 recurrence, RandomState, fixed seed and alpha passthrough."""
    from numpy.random import RandomState
    arr = x.cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32) / 255
    rgb = arr[..., :3].copy()
    rng = RandomState(seed)
    for beta in np.linspace(.0005, .1, 1000, dtype=np.float32):
        eps = rng.normal(size=rgb.shape[1:]).astype(np.float32)
        rgb = float(np.sqrt(1 - beta)) * rgb + float(np.sqrt(beta)) * eps[None]
    arr[..., :3] = np.rint(np.clip(rgb, 0, 1) * 255) / 255
    return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy()) * 255

