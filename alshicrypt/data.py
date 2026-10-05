from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_DATA = Path("deprecated/alshicrypt-multimodal/pokemon")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**63 - 1)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def make_manifest(root: Path, output: Path, seed: int = 17) -> dict:
    if output.exists():
        result = json.loads(output.read_text())
        validate_manifest(result, root)
        return result
    files = sorted(root.rglob("*.png"))
    if not files:
        raise ValueError(f"No PNG files in {root}")
    parent = {p.stem: p.stem for p in files}
    if len(parent) != len(files):
        raise ValueError("Ambiguous duplicate image names")

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    rows, hashes, variants = [], {}, {}
    for path in files:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGBA"))
        if arr.shape != (120, 120, 4):
            raise ValueError(f"Expected 120x120 RGBA: {path}")
        pixel_hash = hashlib.sha256(arr.tobytes()).hexdigest()
        if pixel_hash in hashes:
            join(path.stem, hashes[pixel_hash])
        hashes[pixel_hash] = path.stem
        base = path.stem.split("-")[0]
        if base in variants:
            join(path.stem, variants[base])
        variants[base] = path.stem
        rows.append({"id": path.stem, "path": str(path.relative_to(root)), "sha256": digest(path), "pixel_sha256": pixel_hash})
    # Use only documented evolution links; the supplied CSV is incomplete.
    metadata = root / "pokemon.csv"
    if metadata.exists():
        for row in csv.DictReader(metadata.open()):
            if row.get("Name") in parent and row.get("Evolution") in parent:
                join(row["Name"], row["Evolution"])
    groups = {}
    for row in rows:
        row["group"] = find(row["id"])
        groups.setdefault(row["group"], []).append(row)
    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    targets = {"train": .8 * len(rows), "val": .1 * len(rows), "test": .1 * len(rows)}
    counts = dict.fromkeys(targets, 0)
    for key in sorted(keys, key=lambda k: -len(groups[k])):
        split = max(targets, key=lambda s: (targets[s] - counts[s]) / targets[s])
        for row in groups[key]:
            row["split"] = split
        counts[split] += len(groups[key])
    result = {"version": 1, "seed": seed, "counts": counts, "grouping": "decoded duplicates, name variants, and available CSV evolution links (not complete family holdout)", "images": rows}
    save_json(output, result)
    return result


def validate_manifest(manifest: dict, root: Path) -> None:
    seen, groups, pixels = set(), {}, {}
    for row in manifest["images"]:
        if row["id"] in seen or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid manifest membership")
        seen.add(row["id"])
        for key, table in [("group", groups), ("pixel_sha256", pixels)]:
            old = table.setdefault(row[key], row["split"])
            if old != row["split"]:
                raise ValueError("Split leakage")
        if digest(root / row["path"]) != row["sha256"]:
            raise ValueError(f"Dataset changed: {row['path']}")


def load_split(manifest: dict, root: Path, split: str) -> tuple[torch.Tensor, list[str]]:
    rows = [r for r in manifest["images"] if r["split"] == split]
    arrays = []
    for row in rows:
        with Image.open(root / row["path"]) as im:
            arrays.append(np.asarray(im.convert("RGBA")).copy())
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous(), [r["id"] for r in rows]


def training_batch(pokemon: torch.Tensor, regime: str, batch: int, generator: torch.Generator) -> torch.Tensor:
    count = batch if regime == "pokemon" else batch // 2 if regime == "mixed" else 0
    pieces = []
    if count:
        indices = torch.randint(len(pokemon), (count,), generator=generator)
        pieces.append(pokemon[indices])
    if count < batch:
        pieces.append(torch.randint(256, (batch - count, 4, *pokemon.shape[-2:]), generator=generator, dtype=torch.uint8))
    return torch.cat(pieces).float()


def synthetic_set(split: str, count: int = 32, size: int = 120) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed_for("static", split))
    return torch.randint(256, (count, 4, size, size), generator=g, dtype=torch.uint8)


def probes(size: int = 120) -> tuple[torch.Tensor, list[str]]:
    out, names = [], []
    for value in [0, 1, 127, 128, 254, 255]:
        out.append(torch.full((4, size, size), value, dtype=torch.uint8))
        names.append(f"solid_{value}")
    ramp = torch.linspace(0, 255, size).round().to(torch.uint8)
    out.append(ramp[None, None, :].expand(4, size, size).clone())
    names.append("ramp")
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    out.append((((xx + yy) % 2) * 255).to(torch.uint8)[None].expand(4, -1, -1).clone())
    names.append("checkerboard")
    impulse = torch.zeros(4, size, size, dtype=torch.uint8)
    impulse[:, size // 2, size // 2] = 255
    out.append(impulse)
    names.append("impulse")
    return torch.stack(out), names

