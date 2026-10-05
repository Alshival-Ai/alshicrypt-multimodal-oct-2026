"""Render frozen test examples without changing or retraining the experiment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alshicrypt import processes as process
from alshicrypt.data import DEFAULT_DATA, digest, load_split, seed_for, validate_manifest, save_json
from alshicrypt.models import make_model
from alshicrypt.train import environment, setup, source_hash

RUN = "extended/gaussian_coupling_pokemon_fresh_s17"
COUNT = 25
BATCH = 4
INK = "#172b46"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def display_rgba(pixels):
    yy, xx = np.indices(pixels.shape[:2])
    checker = np.where((xx // 8 + yy // 8) % 2, 226, 246).astype(np.uint8)
    background = Image.fromarray(np.repeat(checker[..., None], 3, axis=2)).convert("RGBA")
    return Image.alpha_composite(background, Image.fromarray(pixels)).convert("RGB")


def card(record, panels, size):
    width, height = 3 * size + 64, size + 122
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((16, 12), f"{record['index']:02d}  {record['id']}", fill=INK, font=font(18, True))
    for column, (label, panel) in enumerate(zip(("Original", "Encrypted*", "Decrypted"), panels)):
        left = 16 + column * (size + 16)
        draw.text((left, 43), label, fill=INK, font=font(13, True))
        out.paste(panel.resize((size, size), Image.Resampling.NEAREST), (left, 65))
    exact = "yes" if record["exact_rgba_bytes"] else "NO"
    draw.text((16, size + 76), f"Float MAE: {record['mae']:.2e}  |  Exact bytes: {exact}", fill=INK, font=font(12))
    draw.text((16, size + 98), "*RGB preview only; decoding uses the float RGBA payload.", fill="#52647b", font=font(11))
    return out


def sheet(records, panels, columns, size, title):
    cards = [card(r, p, size) for r, p in zip(records, panels)]
    cw, ch = cards[0].size
    rows = (len(cards) + columns - 1) // columns
    width = 48 + columns * cw + (columns - 1) * 12
    height = 100 + rows * ch + (rows - 1) * 12 + 72
    out = Image.new("RGB", (width, height), "#edf1f7")
    draw = ImageDraw.Draw(out)
    draw.text((24, 20), title, fill=INK, font=font(24, True))
    draw.text((24, 57), "Held-out Pokemon | Gaussian coupling | Pokemon-only training | Seed 17", fill=INK, font=font(14))
    for i, tile in enumerate(cards):
        out.paste(tile, (24 + (i % columns) * (cw + 12), 100 + (i // columns) * (ch + 12)))
    y = height - 57
    draw.text((24, y), "Encrypted preview clips RGB to [0,1] and hides transformed alpha for display only.", fill=INK, font=font(13))
    draw.text((24, y + 22), "Recovery uses saved float RGBA + noise. Invertibility is built in; these images do not establish secrecy.", fill=INK, font=font(12))
    return out


@torch.no_grad()
def generate(campaign, output, data):
    selection_path = campaign / "selection.json"
    selection = json.loads(selection_path.read_text())
    manifest_path = campaign / "split.json"
    if selection["source_sha256"] != source_hash():
        raise ValueError("Experiment source differs from frozen selection")
    if selection["split_sha256"] != digest(manifest_path):
        raise ValueError("Split differs from frozen selection")
    entry = next(c for c in selection["candidates"] if c["run"] == RUN)
    checkpoint = campaign / RUN / "best.pt"
    if digest(checkpoint) != entry["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from frozen selection")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, data)
    images, ids = load_split(manifest, data, "test")
    if len(ids) < COUNT:
        raise ValueError("Need at least 25 held-out images")
    previous = json.loads((campaign / RUN / "test.json").read_text())
    if previous["selection_sha256"] != digest(selection_path):
        raise ValueError("Previous test results have different provenance")
    previous = previous["results"]["pokemon"]["per_image"]
    if previous["ids"] != ids:
        raise ValueError("Test image order differs from original evaluation")
    device = setup(17)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = make_model("coupling", "gaussian", saved["config"]["width"]).to(device).eval()
    model.load_state_dict(saved["model"])
    output.mkdir(parents=True, exist_ok=True)
    noise_seed = seed_for("evaluation-noise", "test", "pokemon")
    generator = torch.Generator().manual_seed(noise_seed)
    records, visual_panels = [], []
    source_rows = {r["id"]: r for r in manifest["images"]}
    for start in range(0, COUNT, BATCH):
        # Keep the full original batch at the selection boundary: image 25 was
        # originally evaluated alongside images 26--28, not as a one-item batch.
        raw = images[start:start + BATCH]
        x = (raw.double() / 255).float().to(device)
        noise64 = process.noise_field(tuple(raw.shape), "gaussian", generator)
        noise = noise64.float().to(device)
        encoded = model.encode(x, noise)
        if not torch.isfinite(encoded).all():
            raise FloatingPointError("Nonfinite encoded payload")
        transported, _ = process.serialize_roundtrip(encoded, "gaussian")
        for local in range(min(BATCH, COUNT - start)):
            name = ids[start + local]
            folder = output / name
            folder.mkdir(exist_ok=True)
            np.save(folder / "encrypted.npy", encoded[local].cpu().numpy(), allow_pickle=False)
            np.save(folder / "noise.npy", noise[local].cpu().numpy(), allow_pickle=False)
            transported[local] = torch.from_numpy(np.load(folder / "encrypted.npy", allow_pickle=False)).to(device)
            noise[local] = torch.from_numpy(np.load(folder / "noise.npy", allow_pickle=False)).to(device)
        restored = model.decode(transported, noise)
        if not torch.isfinite(restored).all():
            raise FloatingPointError("Nonfinite reconstruction")
        for local in range(min(BATCH, COUNT - start)):
            index = start + local
            name, folder = ids[index], output / ids[index]
            original = raw[local].permute(1, 2, 0).numpy()
            recovered = restored[local].cpu()
            reconstructed = (recovered * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
            error = (recovered.double() - x[local].cpu().double()).abs()
            record = {
                "index": index + 1, "id": name, "split": "test",
                "mae": float(error.mean()), "max_error": float(error.max()),
                "exact_rgba_bytes": bool(np.array_equal(original, reconstructed)),
                "mismatched_rgba_bytes": int(np.count_nonzero(original != reconstructed)),
                "source_sha256": source_rows[name]["sha256"],
                "payload_sha256": digest(folder / "encrypted.npy"),
                "noise_sha256": digest(folder / "noise.npy"),
            }
            # Match the already-published test run, with a small float tolerance.
            for key in ("mae", "max_error"):
                prior = previous["metrics"]["roundtrip"][key][index]
                if not np.isclose(record[key], prior, rtol=1e-5, atol=1e-10):
                    raise ValueError(f"{name}: {key} does not reproduce the frozen test result")
            preview = (encoded[local, :3].cpu().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            Image.fromarray(original).save(folder / "original.png")
            Image.fromarray(reconstructed).save(folder / "decrypted.png")
            Image.fromarray(preview).save(folder / "encrypted-preview.png")
            np.save(folder / "decrypted.npy", recovered.numpy(), allow_pickle=False)
            panels = (display_rgba(original), Image.fromarray(preview), display_rgba(reconstructed))
            card(record, panels, 240).save(folder / "triplet.png")
            records.append(record)
            visual_panels.append(panels)
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "run": RUN, "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "checkpoint_step": saved["step"], "selection_sha256": digest(selection_path),
        "split_sha256": digest(manifest_path), "environment": environment(),
        "renderer_sha256": digest(Path(__file__)), "sampling": "first 25 test images in frozen manifest order",
        "evaluation_noise_seed": noise_seed, "evaluation_batch_size": BATCH,
        "noise_generation": "CPU torch.randn float64, variance 0.75; float32 conditioning; batches of four",
        "payload_format": "NumPy float32 CHW RGBA; full noise field saved separately",
        "preview": "RGB clipped to [0,1], opaque; not used for decoding",
        "model_trained_alpha": True, "records": records,
        "mean_mae": float(np.mean([r["mae"] for r in records])),
        "max_error": max(r["max_error"] for r in records),
        "exact_image_count": sum(r["exact_rgba_bytes"] for r in records),
    }
    save_json(output / "manifest.json", metadata)
    sheet(records, visual_panels, 5, 120, "25 held-out examples: Original / Encrypted / Decrypted").save(output / "overview.png")
    for start in range(0, COUNT, 5):
        page = sheet(records[start:start + 5], visual_panels[start:start + 5], 1, 240,
                     f"Original / Encrypted / Decrypted — examples {start + 1}–{start + 5}")
        page.save(output / f"sheet-{start // 5 + 1:02d}.png")
    (output / "README.txt").write_text(
        "25 held-out Pokemon: Gaussian coupling, Pokemon-only training, seed 17.\n"
        "Open overview.png, sheet-01.png through sheet-05.png, or each named triplet.png.\n"
        "Images were selected by manifest order, without filtering for reconstruction quality.\n\n"
        "Encrypted PNGs are opaque RGB previews clipped to [0,1]. They are not the payload.\n"
        "The actual encrypted.npy contains unclipped float32 RGBA in CHW order. Alpha is transformed.\n"
        "Decoding reloads encrypted.npy and noise.npy; it never uses the preview or the original image.\n"
        "Originals and byte-rounded reconstructions are shown on a checkerboard.\n"
        "metrics.csv reports errors before byte rounding, including transparent/background RGB.\n"
        "Exact RGBA recovery compares decoded pixels, not PNG container bytes.\n"
        "The inverse is built into the architecture; these figures do not demonstrate secrecy or target-map fidelity.\n\n"
        f"Exact byte recovery: {metadata['exact_image_count']}/25 images.\n"
        f"Mean normalized MAE: {metadata['mean_mae']:.8g}; maximum error: {metadata['max_error']:.8g}.\n"
        "Reproduce from repository root: .venv/bin/python scripts/render_gaussian_examples.py\n"
    )
    print(json.dumps({"output": str(output), "count": len(records), "exact_images": metadata["exact_image_count"],
                      "mean_mae": metadata["mean_mae"], "max_error": metadata["max_error"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT / "runs/campaign-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "research/examples/gaussian-pokemon-s17")
    parser.add_argument("--data", type=Path, default=ROOT / DEFAULT_DATA)
    args = parser.parse_args()
    generate(args.campaign, args.output, args.data)
