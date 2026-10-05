from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import save_json


def report(output: Path):
    destination = Path("paper/generated")
    destination.mkdir(parents=True, exist_ok=True)
    # Never carry conclusions or optional artifacts over from another campaign.
    (destination / "conclusions.tex").unlink(missing_ok=True)
    optional = ["split.json", "profiles.json", "promotion.json", "selection.json", "diagnostics.json", "screening_schedule.json", "old-process.pdf", "old-process.png", "resource-notes.json", "noise-cross-validation.json", "runtime-portability.json"]
    for name in optional:
        if not (output / name).exists():
            (destination / name).unlink(missing_ok=True)
    save_json(destination / "campaign.json", {"campaign": str(output), "test_opened": (output / "test_opened.json").exists()})
    rows = []
    curves = []
    metadata = []
    for path in sorted(output.glob("*/*/result.json")):
        r = json.loads(path.read_text())
        c = r["config"]
        metadata.append({"phase": path.parent.parent.name, "run": path.parent.name, "config": c, "untrained": r["untrained"], "environment": json.loads((path.parent / "environment.json").read_text())})
        for point in json.loads((path.parent / "history.json").read_text()):
            curves.append({"phase": path.parent.parent.name, "run": path.parent.name, "step": point["step"], "forward_loss": point["forward_loss"], "reverse_loss": point["reverse_loss"], "domains": {d: {"forward_mae": m["forward"]["mae"], "reverse_mae": m["reverse"]["mae"], "roundtrip_mae": m["roundtrip"]["mae"]} for d, m in point["validation"].items()}})
        for split, evaluated in [("validation", r["validation"])]:
            for domain, values in evaluated.items():
                rows.append({"phase": path.parent.parent.name, "run": path.parent.name, "track": c["track"], "family": c["family"], "regime": c["regime"], "seed": c["seed"], "split": split, "domain": domain, "steps": c["steps"], "noise_mode": c["noise_mode"], "forward_mae": values["forward"]["mae"], "reverse_mae": values["reverse"]["mae"], "roundtrip_mae": values["roundtrip"]["mae"], "exact_image_rate": values["roundtrip"].get("exact_image_rate"), "forward_byte_accuracy": values["forward"].get("byte_accuracy"), "parameters": r["parameters"], "training_seconds": r["training_seconds"], "peak_vram_mib": r["peak_allocated_bytes"] / 2**20})
        test_file = path.parent / "test.json"
        if test_file.exists():
            test = json.loads(test_file.read_text())["results"]
            for domain, values in test.items():
                template = next(row for row in reversed(rows) if row["run"] == path.parent.name and row["domain"] == domain)
                rows.append({**template, "split": "test", "forward_mae": values["forward"]["mae"], "reverse_mae": values["reverse"]["mae"], "roundtrip_mae": values["roundtrip"]["mae"], "exact_image_rate": values["roundtrip"].get("exact_image_rate"), "forward_byte_accuracy": values["forward"].get("byte_accuracy")})
    if not rows:
        raise ValueError("No completed runs to report")
    with (destination / "results.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(destination / "results.json", rows)
    save_json(destination / "learning-curves.json", curves)
    save_json(destination / "run-metadata.json", metadata)
    screening_table = [r"\begin{tabular}{lllrrr}", r"\toprule", r"Track & Model & Data & Target MAE sum & Round-trip MAE & Target byte \% \\", r"\midrule"]
    for row in rows:
        if row["phase"] != "screening" or row["split"] != "validation" or row["domain"] != "pokemon":
            continue
        accuracy = "--" if row["forward_byte_accuracy"] is None else f"{100 * row['forward_byte_accuracy']:.1f}"
        screening_table.append(f"{row['track']} & {row['family']} & {row['regime']} & {row['forward_mae'] + row['reverse_mae']:.4g} & {row['roundtrip_mae']:.4g} & {accuracy} " + r"\\")
    screening_table += [r"\bottomrule", r"\end{tabular}"]
    (destination / "screening-table.tex").write_text("\n".join(screening_table) + "\n")
    for name in optional:
        if (output / name).exists():
            shutil.copyfile(output / name, destination / name)
    tested = [r for r in rows if r["phase"] == "extended" and r["split"] == "test" and r["domain"] == "pokemon"]
    source = tested or [r for r in rows if r["phase"] == "screening" and r["split"] == "validation" and r["domain"] == "pokemon"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, track in zip(axes, ["byte", "gaussian"]):
        selected = [r for r in source if r["track"] == track]
        if not selected:
            ax.text(.5, .5, "No completed runs yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(track.title())
            continue
        families = sorted({r["family"] for r in selected})
        for i, family in enumerate(families):
            means, deviations = [], []
            for regime in ["pokemon", "static", "mixed"]:
                values = [r["forward_mae"] + r["reverse_mae"] for r in selected if r["family"] == family and r["regime"] == regime]
                means.append(np.mean(values) if values else np.nan)
                deviations.append(np.std(values, ddof=1) if len(values) > 1 else 0)
            ax.errorbar(np.arange(3) + i * .06, means, yerr=deviations, marker="o", capsize=3, label=family)
        ax.set_xticks(range(3), ["Pokémon", "Static", "Mixed"])
        ax.set_title(f"{track.title()} target learning")
        ax.set_ylabel("Forward + reverse normalized MAE (lower is better)")
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(destination / "comparison.png", dpi=180)
    fig.savefig(destination / "comparison.pdf")
    plt.close(fig)
    configs = {(m["phase"], m["run"]): m["config"] for m in metadata}
    static_curves = [p for p in curves if p["phase"] == "extended" and configs[(p["phase"], p["run"])]["regime"] == "static"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, track in zip(axes, ["byte", "gaussian"]):
        points = [p for p in static_curves if configs[(p["phase"], p["run"])]["track"] == track]
        steps = sorted({p["step"] for p in points})
        for domain, label in [("pokemon", "Pokémon validation"), ("static", "Static validation"), ("probes", "Structured probes")]:
            if not steps:
                continue
            means, sd = [], []
            for step in steps:
                values = [p["domains"][domain]["forward_mae"] + p["domains"][domain]["reverse_mae"] for p in points if p["step"] == step]
                means.append(np.mean(values))
                sd.append(np.std(values, ddof=1) if len(values) > 1 else 0.)
            means, sd = np.asarray(means), np.asarray(sd)
            line, = ax.plot(steps, means, label=label)
            ax.fill_between(steps, np.maximum(means - sd, 0), means + sd, color=line.get_color(), alpha=.12)
        ax.set_title(f"{track.title()}: static-only training")
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("Forward + reverse target MAE")
        ax.grid(alpha=.2)
        if points:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, "No repeated curves yet", ha="center", transform=ax.transAxes)
    fig.savefig(destination / "static-transfer.pdf")
    fig.savefig(destination / "static-transfer.png", dpi=180)
    plt.close(fig)
    table = [r"\begin{tabular}{lllrrr}", r"\toprule", r"Track & Model & Data & Forward MAE & Reverse MAE & Round-trip MAE \\", r"\midrule"]
    summary = []
    keys = sorted({(r["track"], r["family"], r["regime"]) for r in source})
    for track, family, regime in keys:
        group = [r for r in source if (r["track"], r["family"], r["regime"]) == (track, family, regime)]
        vals = [np.mean([r[k] for r in group]) for k in ["forward_mae", "reverse_mae", "roundtrip_mae"]]
        std = [float(np.std([r[k] for r in group], ddof=1)) if len(group) > 1 else 0 for k in ["forward_mae", "reverse_mae", "roundtrip_mae"]]
        table.append(f"{track} & {family} & {regime} & {vals[0]:.4g} & {vals[1]:.4g} & {vals[2]:.4g} " + r"\\")
        summary.append({"track": track, "family": family, "regime": regime, "seeds": len(group), "mean": dict(zip(["forward_mae", "reverse_mae", "roundtrip_mae"], map(float, vals))), "std": dict(zip(["forward_mae", "reverse_mae", "roundtrip_mae"], std))})
    table += [r"\bottomrule", r"\end{tabular}"]
    (destination / "table.tex").write_text("\n".join(table) + "\n")
    save_json(destination / "summary.json", {"split": "test" if tested else "validation", "groups": summary})
    stage = "Frozen test evaluation" if tested else "Preliminary validation screening"
    (destination / "status.tex").write_text(r"\newcommand{\ResultStage}{" + stage + "}\n")
    if (output / "selection.json").exists() and tested:
        selection = json.loads((output / "selection.json").read_text())
        if len(tested) != len(selection["candidates"]):
            (destination / "status.tex").write_text(r"\newcommand{\ResultStage}{Partial frozen test evaluation}" + "\n")
            return summary
        track = selection["primary_track"]
        family = selection["families"][track]
        regime = selection["winning_regimes"][track]
        paragraphs = [r"\subsection{Selection and transfer}",
            f"Validation selected the {family} architecture with {regime} training for the {track} track as the main approach. "
            "This selection was frozen before test evaluation. Both transformation tracks and all training regimes remain reported."]
        for t in ["byte", "gaussian"]:
            groups = {s["regime"]: s for s in summary if s["track"] == t}
            target = {key: val["mean"]["forward_mae"] + val["mean"]["reverse_mae"] for key, val in groups.items()}
            paragraphs.append(f"For {t}, the mean held-out forward-plus-reverse MAE was {target['pokemon']:.5g} with sprite training, "
                f"{target['static']:.5g} with static training, and {target['mixed']:.5g} with mixed training. "
                "These descriptive differences concern target learning; exact self-inversion alone is not evidence of transfer.")
            if t == "byte":
                parts = []
                for data_regime in ["pokemon", "static", "mixed"]:
                    values = [100 * r["forward_byte_accuracy"] for r in tested if r["track"] == "byte" and r["regime"] == data_regime]
                    parts.append(f"{data_regime}: ${np.mean(values):.2f}\\pm{np.std(values, ddof=1):.2f}$\\%")
                paragraphs.append("Byte forward-target accuracy (mean and sample standard deviation across training seeds) was " + "; ".join(parts) + ". These percentages measure agreement with the prescribed teacher, not self-round-trip recovery.")
        gaussian_max = []
        for candidate in selection["candidates"]:
            if candidate["config"]["track"] == "gaussian":
                result = json.loads((output / candidate["run"] / "test.json").read_text())["results"]
                gaussian_max.extend(m["roundtrip"]["max_error"] for m in result.values())
        if gaussian_max:
            worst = max(gaussian_max)
            paragraphs.append(f"The largest Gaussian learned-round-trip error across retained test runs and domains was {worst:.5g}. "
                + ("This is comfortably below half an 8-bit step ($1/510$), including input float-rounding error, so rounding reconstructed values to the nearest original byte recovers the input byte grid. This is a numerical recovery property, not evidence of target learning."
                   if worst + 1e-7 < 1 / 510 else "Recovery of original bytes cannot be guaranteed merely by rounding at this error level."))
        paragraphs += [r"\subsection{Costs and diagnostics}"]
        min_steps = min(r["steps"] for r in tested)
        max_steps = max(r["steps"] for r in tested)
        paragraphs.append(f"Extended runs used {min_steps}--{max_steps} updates ({min_steps/100:g}--{max_steps/100:g} defined epochs), "
            "with an effective batch size of 16. The retained coupling networks have 44,296 parameters; "
            "the independent byte and Gaussian pairs have 294,208 and 226,888 parameters respectively.")
        if (output / "diagnostics.json").exists():
            d = json.loads((output / "diagnostics.json").read_text())
            old = d["old_process"]
            paragraphs.append(f"The original process yielded {old['unique_rgb_arrays']} unique RGB arrays among {old['images']} validation images, "
                f"while preserving alpha exactly. Silhouette-only nearest-training-mask retrieval achieved normalized RGB MAE "
                f"{d['alpha_only_nearest_training_mask']['rgb_mae']:.5g}. "
                f"Clipping and quantizing the gentler Gaussian endpoint to PNG produced analytic recovery MAE "
                f"{d['gaussian_png_clipping_and_quantization']['mae']:.5g}; "
                "this illustrates why Gaussian results require an unclipped float payload.")
        if (output / "noise-cross-validation.json").exists():
            matrix = json.loads((output / "noise-cross-validation.json").read_text())
            paragraphs.append(r"\subsection{Noise realization transfer}")
            for t in ["byte", "gaussian"]:
                keys = [f"{t}_{train}_{evaluation}" for train in ["fresh", "fixed"] for evaluation in ["fresh", "fixed"]]
                if not all(key in matrix for key in keys):
                    continue
                def error(key):
                    m = matrix[key]["results"]["pokemon"]
                    return m["forward"]["mae"] + m["reverse"]["mae"]
                paragraphs.append(f"For {t}, the fresh-noise-trained mixed model had validation target MAE sum {error(keys[0]):.5g} on fresh fields and {error(keys[1]):.5g} on the fixed field. "
                    f"The fixed-noise-trained counterpart had errors {error(keys[2]):.5g} on fresh fields and {error(keys[3]):.5g} on the fixed field. "
                    "This single-seed diagnostic crosses training and evaluation noise conditions; it is excluded from architecture selection.")
        (destination / "conclusions.tex").write_text("\n\n".join(paragraphs) + "\n")
    return summary
