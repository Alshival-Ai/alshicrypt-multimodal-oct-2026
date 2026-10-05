from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
from PIL import Image

from alshicrypt.data import digest, load_split, save_json, seed_for, validate_manifest
from alshicrypt.models import make_model
from alshicrypt.processes import serialize_roundtrip
from alshicrypt.train import setup
from scripts.render_gaussian_examples import card, display_rgba, sheet
from .core import ROOT, SEEDS, Schedule, evaluate_domains, load_model, verify_selection


def tex_number(value, digits=3):
    text = f"{value:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return mantissa + r"\times10^{" + str(int(exponent)) + "}"
    return text


@torch.no_grad()
def mew_map_figure(output, data, paper):
    """Illustrate sigma and tau with one actual frozen-model round trip."""
    verify_selection(output)
    manifest = json.loads((output / "split.json").read_text())
    row = next(r for r in manifest["images"] if r["id"] == "mew")
    source = data / row["path"]
    if digest(source) != row["sha256"]:
        raise ValueError("Mew source differs from the frozen dataset")
    with Image.open(source) as im:
        rgba = np.asarray(im.convert("RGBA")).copy()
    device = setup(17)
    checkpoint = output / "seed-17/best.pt"
    model, schedule, saved = load_model(checkpoint, device)
    raw = torch.from_numpy(rgba).permute(2, 0, 1).unsqueeze(0)
    x = (raw.double() / 255).float().to(device)
    noise_seed = seed_for("paper-mew-learned-map", schedule.steps)
    noise = schedule.noise(tuple(x.shape), torch.Generator().manual_seed(noise_seed)).float().to(device)
    encoded = model.encode(x, noise)
    transmitted, _ = serialize_roundtrip(encoded, "gaussian")
    decoded = model.decode(transmitted, noise)
    recovered = (decoded * 255).round().clamp(0, 255).to(torch.uint8)
    exact = torch.equal(recovered.cpu(), raw)
    if not exact or not torch.isfinite(decoded).all():
        raise ArithmeticError("Mew illustration did not recover all original RGBA bytes")
    preview = (encoded[0, :3].clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    restored_rgba = recovered[0].permute(1, 2, 0).cpu().numpy()
    panels = [display_rgba(rgba), Image.fromarray(preview), display_rgba(restored_rgba)]
    fig = plt.figure(figsize=(9.5, 2.8))
    for left, panel, title in zip((.035, .39, .745), panels, ("Mew", "Transformed", "Recovered Mew")):
        ax = fig.add_axes([left, .08, .22, .78])
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(title, fontsize=12, color="#233247")
        ax.axis("off")
    arrows = fig.add_axes([0, 0, 1, 1], frameon=False)
    arrows.set(xlim=(0, 1), ylim=(0, 1))
    arrows.axis("off")
    for start, end, symbol in ((.27, .375, r"$\sigma$"), (.625, .73, r"$\tau$")):
        arrows.annotate("", xy=(end, .47), xytext=(start, .47),
                        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#233247"})
        arrows.text((start + end) / 2, .56, symbol, ha="center", va="center", fontsize=23, color="#233247")
    destination = paper / "generated"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "mew-learned-map.pdf")
    fig.savefig(destination / "mew-learned-map.png", dpi=220)
    plt.close(fig)
    arrays = destination / "mew-learned-map.npz"
    np.savez_compressed(arrays, original=raw[0].numpy(), encoded=transmitted[0].cpu().numpy(),
                        noise=noise[0].cpu().numpy(), decoded=decoded[0].cpu().numpy(),
                        recovered=recovered[0].cpu().numpy())
    save_json(destination / "mew-learned-map.json", {
        "image": "mew", "dataset_split": row["split"], "source": row["path"],
        "source_sha256": digest(source), "checkpoint_sha256": digest(checkpoint),
        "model_seed": 17, "update": saved["step"], "stochastic_steps": schedule.steps,
        "noise_seed": noise_seed, "noise_generator": "CPU torch.randn float64, scaled and cast to float32",
        "device": str(device), "tensor_layout": "CHW", "channels": "RGBA",
        "roundtrip_mae": (decoded - x).abs().mean().item(),
        "roundtrip_max_error": (decoded - x).abs().max().item(), "exact_rgba_bytes": exact,
        "arrays_sha256": digest(arrays),
        "purpose": "opening illustration using a training image; separate from test results",
        "display": "original and recovered RGBA on checkerboard; transformed clipped RGB preview"})


def geometry_figure(paper):
    """A two-coordinate polynomial coupling and its explicit inverse."""
    destination = paper / "generated"
    destination.mkdir(parents=True, exist_ok=True)

    def forward(points):
        u, v = points
        return np.array([u, .5 * v + .3 * u ** 2])

    def inverse(points):
        u, v = points
        return np.array([u, 2 * (v - .3 * u ** 2)])

    axis = np.linspace(-1, 1, 201)
    lines = []
    for level in np.linspace(-1, 1, 9):
        lines.extend([np.array([axis, np.full_like(axis, level)]),
                      np.array([np.full_like(axis, level), axis])])
    for line in lines:
        if not np.allclose(inverse(forward(line)), line, atol=1e-14, rtol=0):
            raise ArithmeticError("Geometric example failed its inverse check")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    points = np.array([[-.6, .6], [-.4, .4]])
    for ax, transform, title, xlabel, ylabel in zip(
        axes, (lambda x: x, forward, lambda x: inverse(forward(x))),
        ("Original coordinates", "After coupling", "After inverse"),
        (r"$u$", r"$u'$", r"$u$"), (r"$v$", r"$v'$", r"$v$"),
    ):
        for i, line in enumerate(lines):
            mapped = transform(line)
            ax.plot(*mapped, color="#3976aa" if i % 2 else "#9bafc0", lw=.9)
        mapped = transform(points)
        ax.scatter(*mapped, c=["#a02e87", "#de7618"], s=35, zorder=3)
        for j, name in enumerate(("A", "B")):
            ax.annotate(name, mapped[:, j], xytext=(5, 5), textcoords="offset points", fontsize=10)
        ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), xlabel=xlabel, ylabel=ylabel,
               xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(destination / "coupling-geometry.pdf")
    fig.savefig(destination / "coupling-geometry.png", dpi=220)
    plt.close(fig)
    save_json(destination / "coupling-geometry.json", {
        "purpose": "two-coordinate mathematical illustration",
        "forward": "(u, v) -> (u, 0.5*v + 0.3*u**2)",
        "inverse": "(u_prime, v_prime) -> (u_prime, 2*(v_prime - 0.3*u_prime**2))",
        "determinant": .5, "grid_range": [-1, 1], "grid_lines_per_axis": 9,
        "points": points.T.tolist()})


def process_figure(paper):
    """Two evaluation paths, with the same original and noise in both."""
    destination = paper / "generated"
    destination.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98)
    ax.set(xlim=(0, 11), ylim=(0, 4))
    ax.axis("off")
    ink = "#233247"
    ax.text(5.5, 3.78, "Same original x and saved noise n in both paths", ha="center",
            fontsize=14, color=ink, weight="bold")
    for y, title, color, labels in (
        (2.45, "PRESCRIBED TARGET", "#e9f2fc",
         [r"$T(x,n)=0.25x+n$", r"Target $y$", r"$\tau_\theta(y,n)$", r"$\widetilde{x}$"]),
        (.9, "LEARNED TRANSFORMATION", "#fff0df",
         [r"$\sigma_\theta(x,n)$", r"Saved output $z$", r"$\tau_\theta(z,n)$", r"$\widehat{x}$"]),
    ):
        ax.text(.12, y + .62, title, fontsize=11, color=ink, weight="bold")
        ax.text(.35, y, r"$x$", fontsize=18, va="center", color=ink)
        for x, width, label in zip((1, 4.05, 6.6, 9.25), (2.35, 1.85, 2, 1.25), labels):
            ax.add_patch(FancyBboxPatch((x, y - .35), width, .7,
                         boxstyle="round,pad=0.05", facecolor=color, edgecolor="#bac5d1"))
            ax.text(x + width / 2, y, label, fontsize=14, ha="center", va="center", color=ink)
        for start, end in ((.62, .94), (3.42, 3.98), (5.97, 6.53), (8.67, 9.18)):
            ax.annotate("", xy=(end, y), xytext=(start, y),
                        arrowprops={"arrowstyle": "->", "color": ink})
    ax.annotate("", xy=(4.97, 2.05), xytext=(4.97, 1.3),
                arrowprops={"arrowstyle": "<->", "color": ink, "linestyle": "dashed"})
    ax.text(4.72, 1.67, "Forward error", ha="right", va="center", fontsize=11, color=ink)
    ax.text(8.7, 1.91, r"Reverse error: compare $\widetilde{x}$ with $x$", ha="center", fontsize=11, color=ink)
    ax.text(8.7, .34, r"Round-trip error: compare $\widehat{x}$ with $x$", ha="center", fontsize=11, color=ink)
    fig.savefig(destination / "process-overview.pdf")
    fig.savefig(destination / "process-overview.png", dpi=220)
    plt.close(fig)


def saved_result_assets(output, paper):
    """Small teaching assets from frozen results; no inference or new evaluation."""
    verify_selection(output)
    destination = paper / "generated"
    destination.mkdir(parents=True, exist_ok=True)
    table = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Model state & Forward MAE & Reverse MAE & \shortstack{Exact RGBA\\images (\%)} \\", r"\midrule"]
    records = []
    for state, label in (("untrained", "Before training"), ("validation", "After training")):
        values = []
        for seed in SEEDS:
            result = json.loads((output / f"seed-{seed}/result.json").read_text())
            row = result[state]["pokemon"]
            values.append(row)
            records.append({"seed": seed, "state": state, "split": "validation", "count": row["count"],
                            "forward_mae": row["forward"]["mae"], "reverse_mae": row["reverse"]["mae"],
                            "exact_rgba_image_rate": row["exact_rgba_image_rate"]})
        forward = np.mean([v["forward"]["mae"] for v in values])
        reverse = np.mean([v["reverse"]["mae"] for v in values])
        exact = np.mean([v["exact_rgba_image_rate"] for v in values])
        table.append(f"{label} & {forward:.5f} & {reverse:.5f} & {100 * exact:.1f} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (destination / "training-control-table.tex").write_text("\n".join(table) + "\n")
    save_json(destination / "training-control.json", records)
    table = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"Seed & Best update & Forward MAE & Reverse MAE & Round-trip MAE & \shortstack{Exact RGBA\\images (\%)} \\", r"\midrule"]
    for seed in SEEDS:
        folder = output / f"seed-{seed}"
        tested = json.loads((folder / "test.json").read_text())
        if tested["selection_sha256"] != digest(output / "selection.json"):
            raise ValueError("Test results differ from frozen selection")
        row = tested["results"]["pokemon"]
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        table.append(f"{seed} & {checkpoint['step']} & {row['forward']['mae']:.5f} & "
                     f"{row['reverse']['mae']:.5f} & ${tex_number(row['roundtrip']['mae'])}$ & "
                     f"{100 * row['exact_rgba_image_rate']:.1f} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (destination / "test-table.tex").write_text("\n".join(table) + "\n")
    manifest = json.loads((paper / "examples/manifest.json").read_text())
    record = manifest["records"][0]
    directory = paper / "examples" / record["id"]
    with Image.open(directory / "original.png") as source, Image.open(directory / "decrypted.png") as decoded:
        original, restored = np.asarray(source.convert("RGBA")), np.asarray(decoded.convert("RGBA"))
    assert np.array_equal(original, restored) == record["exact_rgba_bytes"]
    with Image.open(directory / "encrypted-preview.png") as preview:
        panels = [display_rgba(original), preview.copy(), display_rgba(restored)]
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)
    for ax, panel, title in zip(axes, panels, ("Original", "Encrypted RGB preview", "Decrypted")):
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.savefig(destination / "example-single.pdf")
    fig.savefig(destination / "example-single.png", dpi=220)
    plt.close(fig)
    (destination / "example-single-caption.tex").write_text(
        f"{record['id'].capitalize()}, the first test image in manifest order (seed 17). "
        f"Float round-trip MAE is ${tex_number(record['mae'])}$; all RGBA bytes are recovered exactly after rounding.\n")


def stochastic_figure(data, paper):
    """One actual forward trajectory, with clipping confined to display copies."""
    destination = paper / "generated"
    destination.mkdir(parents=True, exist_ok=True)
    source = data / "images/mew.png"
    with Image.open(source) as image:
        rgba = np.asarray(image.convert("RGBA")).copy()
    x = torch.from_numpy(rgba).permute(2, 0, 1).double() / 255
    schedule = Schedule()
    seed = seed_for("paper-mew-stochastic-trajectory", schedule.steps)
    generator = torch.Generator().manual_seed(seed)
    state, accumulated = x.clone(), torch.zeros_like(x)
    times = [0, 1, 2, 5, 16, 32]
    states = {0: state.clone()}
    for step in range(1, schedule.steps + 1):
        increment = torch.randn(state.shape, generator=generator, dtype=torch.float64)
        state = (1 - schedule.beta) ** .5 * state + schedule.beta ** .5 * increment
        accumulated = (1 - schedule.beta) ** .5 * accumulated + schedule.beta ** .5 * increment
        coefficient = (1 - schedule.beta) ** (step / 2)
        if not torch.allclose(state, coefficient * x + accumulated, atol=1e-14, rtol=0):
            raise ArithmeticError("Illustrated trajectory disagrees with endpoint identity")
        if step in times:
            states[step] = state.clone()
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.7), constrained_layout=True)
    records = []
    for ax, step in zip(axes.flat, times):
        coefficient = (1 - schedule.beta) ** (step / 2)
        preview = (states[step][:3].clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        display = display_rgba(rgba) if step == 0 else Image.fromarray(preview)
        ax.imshow(display, interpolation="nearest")
        ax.set_title("Original (t = 0)" if step == 0 else f"Step {step}", fontsize=13, weight="bold")
        ax.set_xlabel(f"Signal coefficient: {coefficient:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        records.append({"step": step, "signal_coefficient": coefficient,
                        "noise_variance": 1 - coefficient ** 2})
    fig.savefig(destination / "mew-stochastic-process.pdf")
    fig.savefig(destination / "mew-stochastic-process.png", dpi=220)
    plt.close(fig)
    np.savez_compressed(destination / "mew-stochastic-states.npz",
                        steps=np.asarray(times), states=np.stack([states[t].numpy() for t in times]))
    save_json(destination / "mew-stochastic-process.json", {
        "source": str(source), "source_sha256": digest(source), "image": "mew",
        "purpose": "theory illustration, not a reconstruction evaluation", "dataset_split": "train",
        "seed": seed, "generator": "CPU torch.Generator; torch.randn float64",
        "beta": schedule.beta, "channels": "RGBA", "tensor_layout": "CHW",
        "trajectory": "one shared path of 32 independent Gaussian increments; no intermediate clipping",
        "display": "Original RGBA on checkerboard; later panels opaque clipped RGB previews",
        "snapshots": records})


def coupling_figure(data, paper):
    """Explain the channel partition and inverse; no model outputs are depicted."""
    destination = paper / "generated"
    destination.mkdir(parents=True, exist_ok=True)
    source = data / "images/mew.png"
    with Image.open(source) as image:
        rgba = np.asarray(image.convert("RGBA")).copy()
    fig, ax = plt.subplots(figsize=(10, 8.1))
    fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.985)
    ax.set(xlim=(0, 10), ylim=(0, 8.1))
    ax.axis("off")
    ink, blue, orange = "#233247", "#e9f2fc", "#fff0df"

    def label(x, y, text, size=12, **kwargs):
        ax.text(x, y, text, fontsize=size, color=ink, va="center", **kwargs)

    def box(x, y, w, h, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                  facecolor=color, edgecolor="#bac5d1", linewidth=.8))

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": ink, "lw": 1.4})

    label(.12, 7.87, "A. Split the channels; each group covers the whole image", 14, weight="bold")
    ax.imshow(display_rgba(rgba), extent=(.2, 1.65, 5.94, 7.39), interpolation="nearest")
    label(.925, 5.73, "Mew (RGBA)", ha="center")
    arrow((1.78, 6.64), (2.5, 6.64))
    for x, indices, title, color in ((2.7, (0, 1), r"$u$: red + green", blue),
                                     (6.25, (2, 3), r"$v$: blue + alpha", orange)):
        box(x, 5.51, 3.2, 1.98, color)
        label(x + 1.6, 7.26, title, 13, ha="center", weight="bold")
        for offset, channel in enumerate(indices):
            left = x + .2 + offset * 1.55
            ax.imshow(rgba[:, :, channel], cmap="gray", vmin=0, vmax=255,
                      extent=(left, left + 1.25, 5.89, 7.04), interpolation="nearest", zorder=2)
            label(left + .625, 5.68, ("Red", "Green", "Blue", "Alpha")[channel], 11, ha="center")
    label(5, 5.2, "Channel values shown in grayscale: black = 0, white = 1. Alpha is a data channel.",
          11, ha="center")

    label(.12, 4.79, "B. One block: keep u available to reproduce the adjustment", 14, weight="bold")
    for x, title, color in ((.15, "FORWARD", blue), (5.15, "REVERSE", orange)):
        box(x, 2.28, 4.65, 2.22, color)
        label(x + .18, 4.24, title, 12, weight="bold")
    label(.48, 3.87, r"Keep $u'=u$; compute $t=t_\theta(u,n)$.", 13)
    label(5.48, 3.87, r"Keep $u=u'$; recompute $t=t_\theta(u',n)$.", 13)
    label(2.48, 3.28, r"$v\ \longrightarrow\ 0.5v+t\ =\ v'$", 19, ha="center")
    label(7.48, 3.28, r"$v'\ \longrightarrow\ (v'-t)/0.5\ =\ v$", 19, ha="center")
    label(2.48, 2.74, r"Example: $0.5\times0.8+0.3=0.7$", 12, ha="center")
    label(7.48, 2.74, r"Undo: $(0.7-0.3)/0.5=0.8$", 12, ha="center")
    label(5, 2, "Same network + unchanged group + same saved noise = same adjustment.", 12, ha="center")

    label(.12, 1.57, "C. Alternate groups across the four network blocks", 14, weight="bold")
    for i in range(4):
        x = .2 + 2.5 * i
        box(x, .52, 2.05, .77, blue if i % 2 == 0 else orange)
        held, changed = ("RG", "B + alpha") if i % 2 == 0 else ("B + alpha", "RG")
        label(x + 1.025, 1.09, f"Block {i + 1}: keep {held}", 11, ha="center")
        label(x + 1.025, .74, f"Update {changed}", 11, ha="center", weight="bold")
        if i < 3:
            arrow((x + 2.12, .9), (x + 2.4, .9))
    label(5, .17, "Encode: 1 → 2 → 3 → 4     |     Decode: 4 → 3 → 2 → 1", 12, ha="center")
    fig.savefig(destination / "mew-coupling-diagram.pdf")
    fig.savefig(destination / "mew-coupling-diagram.png", dpi=220)
    plt.close(fig)
    save_json(destination / "mew-coupling-diagram.json", {
        "source": str(source), "source_sha256": digest(source),
        "purpose": "conceptual channel-partition and inversion diagram; no model predictions",
        "channel_groups": [["red", "green"], ["blue", "alpha"]],
        "channel_display": "individual original channels in grayscale, fixed range [0, 255]",
        "scale": .5, "scalar_example": {"v": .8, "shift": .3, "v_prime": .7},
        "forward_block_order": [1, 2, 3, 4], "reverse_block_order": [4, 3, 2, 1]})


def figure_triplets(records, panels, destination):
    fig, axes = plt.subplots(len(records), 3, figsize=(8, 9), constrained_layout=True)
    for row, (record, images) in enumerate(zip(records, panels)):
        for column, image in enumerate(images):
            axes[row, column].imshow(image, interpolation="nearest")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(record["id"], fontsize=9)
        axes[row, 2].set_xlabel(f"MAE {record['mae']:.2e}; exact bytes: {record['exact_rgba_bytes']}", fontsize=8)
    for ax, title in zip(axes[0], ("Original", "Encrypted RGB preview", "Decrypted")):
        ax.set_title(title, fontsize=11)
    fig.savefig(destination.with_suffix(".pdf"))
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    plt.close(fig)


@torch.no_grad()
def examples(output, data, paper):
    selected = verify_selection(output)
    manifest = json.loads((output / "split.json").read_text())
    validate_manifest(manifest, data)
    pokemon, ids = load_split(manifest, data, "test")
    source_rows = {r["id"]: r for r in manifest["images"]}
    folder = output / "seed-17"
    device = setup(17)
    model, schedule, saved = load_model(folder / "best.pt", device)
    tested = json.loads((folder / "test.json").read_text())["results"]["pokemon"]["per_image"]
    if tested["ids"] != ids:
        raise ValueError("Example order differs from test")
    generator = torch.Generator().manual_seed(seed_for("evaluation-noise", "test", "pokemon"))
    destination = paper / "examples"
    destination.mkdir(parents=True, exist_ok=True)
    records, panels = [], []
    for start in range(0, 25, 4):
        raw = pokemon[start:start + 4]
        x = (raw.double() / 255).float().to(device)
        noise = schedule.noise(tuple(raw.shape), generator).float().to(device)
        encoded = model.encode(x, noise)
        transported, _ = serialize_roundtrip(encoded, "gaussian")
        for local in range(min(4, 25 - start)):
            target = destination / ids[start + local]
            target.mkdir(exist_ok=True)
            np.save(target / "encrypted.npy", encoded[local].cpu().numpy(), allow_pickle=False)
            np.save(target / "noise.npy", noise[local].cpu().numpy(), allow_pickle=False)
            transported[local] = torch.from_numpy(np.load(target / "encrypted.npy", allow_pickle=False)).to(device)
            noise[local] = torch.from_numpy(np.load(target / "noise.npy", allow_pickle=False)).to(device)
        restored = model.decode(transported, noise)
        for local in range(min(4, 25 - start)):
            index = start + local
            target = destination / ids[index]
            original = raw[local].permute(1, 2, 0).numpy()
            decoded = (restored[local].cpu() * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
            difference = (restored[local].double() - x[local].double()).abs()
            record = {"index": index + 1, "id": ids[index], "split": "test", "mae": float(difference.mean()),
                      "max_error": float(difference.max()), "exact_rgba_bytes": bool(np.array_equal(original, decoded)),
                      "source_sha256": source_rows[ids[index]]["sha256"],
                      "payload_sha256": digest(target / "encrypted.npy"), "noise_sha256": digest(target / "noise.npy")}
            for metric in ("mae", "max_error"):
                if not np.isclose(record[metric], tested["metrics"]["roundtrip"][metric][index], rtol=1e-5, atol=1e-10):
                    raise ValueError(f"Example {ids[index]} differs from test result")
            preview = (encoded[local, :3].cpu().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            Image.fromarray(original).save(target / "original.png")
            Image.fromarray(decoded).save(target / "decrypted.png")
            Image.fromarray(preview).save(target / "encrypted-preview.png")
            np.save(target / "decrypted.npy", restored[local].cpu().numpy(), allow_pickle=False)
            triplet = (display_rgba(original), Image.fromarray(preview), display_rgba(decoded))
            card(record, triplet, 240).save(target / "triplet.png")
            records.append(record)
            panels.append(triplet)
    sheet(records, panels, 5, 120, "32-step model: 25 held-out Pokemon").save(destination / "overview.png")
    for start in range(0, 25, 5):
        number = start // 5 + 1
        subset, images = records[start:start + 5], panels[start:start + 5]
        sheet(subset, images, 1, 240, f"32-step model: examples {start + 1}–{start + 5}").save(destination / f"sheet-{number:02d}.png")
        figure_triplets(subset, images, paper / "generated" / f"examples-{number:02d}")
    save_json(destination / "manifest.json", {"checkpoint_sha256": digest(folder / "best.pt"), "step": saved["step"],
              "selection": selected, "noise_seed": seed_for("evaluation-noise", "test", "pokemon"),
              "batch_size": 4, "schedule": saved["config"]["schedule"], "records": records})
    with (destination / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (destination / "README.txt").write_text(
        "First 25 test images in frozen manifest order; 32-step Gaussian model, seed 17.\n"
        "Decoding uses encrypted.npy and noise.npy (float32 CHW RGBA), not the preview image.\n"
        "Encrypted PNGs clip RGB to [0,1] and hide the transformed alpha for display only.\n"
        "Original/decrypted PNGs retain RGBA. Checkerboards in figures are display backgrounds.\n"
        "MAE and maximum errors are measured before byte rounding; exactness compares all RGBA pixel bytes.\n"
        "These examples establish numerical recovery, not confidentiality.\n")
    return records


def baseline16(data, destination):
    """Post-hoc visibility diagnostic; never update the frozen research outputs."""
    campaign = ROOT / "runs/campaign-v1"
    selected = json.loads((campaign / "selection.json").read_text())
    manifest = json.loads((campaign / "split.json").read_text())
    if digest(campaign / "split.json") != selected["split_sha256"]:
        raise ValueError("Baseline split changed")
    validate_manifest(manifest, data)
    pokemon, ids = load_split(manifest, data, "test")
    device = setup(17)
    rows = []
    for seed in SEEDS:
        name = f"extended/gaussian_coupling_pokemon_fresh_s{seed}"
        frozen = next(r for r in selected["candidates"] if r["run"] == name)
        folder = campaign / name
        if digest(folder / "best.pt") != frozen["checkpoint_sha256"]:
            raise ValueError("Baseline checkpoint changed")
        model = make_model("coupling", "gaussian", 32).to(device).eval()
        saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        results = evaluate_domains(model, pokemon, ids, Schedule(16), device, "test", per_image=True)
        previous = json.loads((folder / "test.json").read_text())["results"]
        for domain in results:
            for endpoint in ("forward", "reverse", "roundtrip"):
                if not np.isclose(results[domain][endpoint]["mae"], previous[domain][endpoint]["mae"], rtol=1e-5, atol=1e-10):
                    raise ValueError("16-step diagnostic does not reproduce published endpoint metrics")
        rows.append({"seed": seed, "checkpoint_sha256": digest(folder / "best.pt"), "results": results})
        del model
        torch.cuda.empty_cache()
    save_json(destination / "baseline16-diagnostic.json", rows)
    return rows


def render(output, data, paper):
    verify_selection(output)
    generated = paper / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    mew_map_figure(output, data, paper)
    stochastic_figure(data, paper)
    coupling_figure(data, paper)
    process_figure(paper)
    geometry_figure(paper)
    rows, full = [], []
    for seed in SEEDS:
        folder = output / f"seed-{seed}"
        saved = json.loads((folder / "result.json").read_text())
        test = json.loads((folder / "test.json").read_text())
        if test["selection_sha256"] != digest(output / "selection.json"):
            raise ValueError("Test results differ from frozen selection")
        best = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        full.append({"seed": seed, "training": saved, "test": test,
                     "history": json.loads((folder / "history.json").read_text()),
                     "environment": json.loads((folder / "environment.json").read_text()), "best_update": best["step"]})
        for domain, metrics in test["results"].items():
            rows.append({"seed": seed, "domain": domain, "best_update": best["step"],
                         "forward_mae": metrics["forward"]["mae"], "reverse_mae": metrics["reverse"]["mae"],
                         "roundtrip_mae": metrics["roundtrip"]["mae"], "max_roundtrip_error": metrics["roundtrip"]["max_error"],
                         "foreground_reverse_mae": metrics["reverse"]["foreground_rgb_mae"],
                         "exact_rgba_image_rate": metrics["exact_rgba_image_rate"],
                         "payload_rgb_correlation": metrics["mean_correlation"]["payload_rgb"],
                         "preview_rgb_correlation": metrics["mean_correlation"]["preview_rgb"],
                         "training_seconds": saved["training_seconds"], "peak_vram_mib": saved["peak_allocated_bytes"] / 2**20})
    with (generated / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(generated / "results.json", full)
    for name in ("protocol.json", "selection.json", "split.json", "test_opened.json"):
        shutil.copyfile(output / name, generated / name)
    examples(output, data, paper)
    saved_result_assets(output, paper)
    old = baseline16(data, generated)
    groups = {}
    for steps, entries in ((16, old), (32, [{"seed": f["seed"], "results": f["test"]["results"]} for f in full])):
        group = [r["results"]["pokemon"] for r in entries]
        groups[str(steps)] = {}
        for key in ("forward", "reverse", "roundtrip"):
            values = [r[key]["mae"] for r in group]
            groups[str(steps)][key] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
        groups[str(steps)]["target_mae_sum"] = float(np.mean([r["forward"]["mae"] + r["reverse"]["mae"] for r in group]))
        groups[str(steps)]["correlation"] = {key: float(np.mean([r["mean_correlation"][key] for r in group])) for key in ("payload_rgb", "preview_rgb")}
        groups[str(steps)]["exact_rgba_image_rate"] = float(np.mean([r["exact_rgba_image_rate"] for r in group]))
    save_json(generated / "comparison.json", groups)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for row in full:
        history = row["history"]
        steps = [p["step"] for p in history]
        axes[0].plot(steps, [p["mean_forward_loss"] + p["mean_reverse_loss"] for p in history], label=f"Seed {row['seed']}")
        axes[1].plot(steps, [p["validation"]["pokemon"]["forward"]["mae"] + p["validation"]["pokemon"]["reverse"]["mae"] for p in history], label=f"Seed {row['seed']}")
    axes[0].set_ylabel("Forward + reverse training MSE")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Validation forward + reverse MAE")
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(generated / "learning-curves.pdf")
    fig.savefig(generated / "learning-curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, kind in zip(axes, ("payload_rgb", "preview_rgb")):
        vals = []
        for entries in (old, [{"results": f["test"]["results"]} for f in full]):
            vals.append([e["results"]["pokemon"]["mean_correlation"][kind] for e in entries])
        ax.bar(["16 steps", "32 steps"], [np.mean(v) for v in vals], yerr=[np.std(v, ddof=1) for v in vals], capsize=4)
        ax.set_ylabel("Mean per-image RGB Pearson correlation")
        ax.set_title("Full float payload" if kind == "payload_rgb" else "Clipped RGB preview")
        ax.grid(axis="y", alpha=.2)
    fig.savefig(generated / "visibility.pdf")
    fig.savefig(generated / "visibility.png", dpi=180)
    plt.close(fig)
    table = [r"\begin{tabular}{rrrrr}", r"\toprule", r"Steps & Signal & Forward MAE & Reverse MAE & Round-trip MAE \\", r"\midrule"]
    for steps in (16, 32):
        group = groups[str(steps)]
        vals = ["$" + tex_number(group[k]['mean']) + r"\pm" + tex_number(group[k]['sd'], 2) + "$" for k in ("forward", "reverse", "roundtrip")]
        table.append(f"{steps} & {Schedule(steps).signal:g} & " + " & ".join(vals) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (generated / "comparison-table.tex").write_text("\n".join(table) + "\n")
    g = groups["32"]
    max_error = max(r["max_roundtrip_error"] for r in rows)
    exact = min(r["exact_rgba_image_rate"] for r in rows)
    train_minutes = [f["training"]["training_seconds"] / 60 for f in full]
    vram = [f["training"]["peak_allocated_bytes"] / 2**20 for f in full]
    macros = {"ForwardMAE": f"{g['forward']['mean']:.5f}", "ReverseMAE": f"{g['reverse']['mean']:.5f}",
              "RoundtripMAE": tex_number(g['roundtrip']['mean']), "WorstError": tex_number(max_error),
              "ExactRate": f"{100*exact:.1f}", "MinMinutes": f"{min(train_minutes):.1f}", "MaxMinutes": f"{max(train_minutes):.1f}",
              "MinVRAM": f"{min(vram):.0f}", "MaxVRAM": f"{max(vram):.0f}",
              "OldTargetMAE": f"{groups['16']['target_mae_sum']:.5f}", "NewTargetMAE": f"{g['target_mae_sum']:.5f}",
              "OldCorrelation": f"{groups['16']['correlation']['preview_rgb']:.4f}",
              "NewCorrelation": f"{g['correlation']['preview_rgb']:.4f}"}
    (generated / "metrics.tex").write_text("\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in macros.items()) + "\n")
    interpretation = "increased" if g["target_mae_sum"] > groups["16"]["target_mae_sum"] else "decreased"
    foreground = np.mean([r["foreground_reverse_mae"] for r in rows if r["domain"] == "pokemon"])
    (generated / "comparison-findings.tex").write_text(
        f"The mean forward-plus-reverse target MAE {interpretation} from \\OldTargetMAE{{}} at 16 steps to \\NewTargetMAE{{}} at 32 steps. "
        f"Separately, forward-target MAE changed from {groups['16']['forward']['mean']:.5f} to {g['forward']['mean']:.5f}, "
        f"while reverse-target MAE changed from {groups['16']['reverse']['mean']:.5f} to {g['reverse']['mean']:.5f}. "
        "The two directions should therefore also be assessed separately.\n")
    (generated / "findings.tex").write_text(
        f"Mean foreground-only RGB reverse-target MAE at 32 steps was {foreground:.5f}. "
        "The minimum exact RGBA image-recovery rate across all three seeds and all three test domains was \\ExactRate\\%. "
        "The largest floating-point round-trip error across those runs was $\\WorstError$. "
        "These measurements describe the evaluated images and runtime; they are not a universal finite-precision guarantee.\n")
    save_json(generated / "summary.json", {"comparison": groups, "worst_roundtrip_error": max_error,
              "minimum_exact_rgba_rate": exact, "training_minutes": train_minutes, "peak_vram_mib": vram})
    print(json.dumps({"rendered": str(paper), "summary": str(generated / "summary.json")}), flush=True)


def build(paper):
    engine = ROOT / ".tools/tectonic"
    if not engine.exists():
        raise RuntimeError("Install Tectonic at .tools/tectonic to compile the paper")
    subprocess.run([str(engine), str(paper / "main.tex"), "--outdir", str(paper)], check=True)
