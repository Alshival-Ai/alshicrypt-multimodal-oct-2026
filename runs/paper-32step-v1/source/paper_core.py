from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from alshicrypt.data import digest, load_split, probes, save_json, seed_for, synthetic_set, training_batch, validate_manifest
from alshicrypt.evaluate import metrics, selection_score, summarize, synchronize
from alshicrypt.models import Coupling
from alshicrypt.processes import serialize_roundtrip
from alshicrypt.train import environment, setup, source_hash as legacy_source_hash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/paper-32step-v1"
DEFAULT_DATA = ROOT / "deprecated/alshicrypt-multimodal/pokemon"
SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class Schedule:
    steps: int = 32
    beta: float = 1 - 0.5 ** (2 / 16)

    def __post_init__(self):
        if self.steps < 1 or not 0 < self.beta < 1:
            raise ValueError("Invalid stochastic schedule")

    @property
    def signal(self):
        return (1 - self.beta) ** (self.steps / 2)

    @property
    def noise_sd(self):
        return math.sqrt(1 - self.signal ** 2)

    def noise(self, shape, generator):
        return torch.randn(shape, generator=generator, dtype=torch.float64) * self.noise_sd

    def forward(self, x, noise):
        return self.signal * x + noise

    def inverse(self, y, noise):
        return (y - noise) / self.signal


class PaperCoupling(Coupling):
    def __init__(self, schedule=Schedule(), width=32):
        super().__init__("gaussian", width, blocks=4)
        self.scale = schedule.signal ** (2 / len(self.layers))


def provenance():
    # Reporting changes cannot change the mathematical experiment; freeze both
    # the new experiment core and every imported legacy experiment module.
    return {"paper_core_sha256": digest(Path(__file__)), "legacy_source_sha256": legacy_source_hash()}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def config(seed, updates=2000):
    return {"seed": seed, "updates": updates, "schedule": asdict(Schedule()), "width": 32,
            "batch_size": 16, "microbatch": 4, "lr": 3e-4, "weight_decay": 1e-5,
            "gradient_clip": 1., "validation_interval": 100, "regime": "pokemon"}


def atomic_checkpoint(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def prepare(output, data):
    source = ROOT / "runs/campaign-v1/split.json"
    if not source.exists():
        source = ROOT / "research/generated/split.json"
    manifest = json.loads(source.read_text())
    validate_manifest(manifest, data)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "split.json"
    if target.exists() and digest(target) != digest(source):
        raise ValueError("Follow-up split differs from original study")
    if not target.exists():
        shutil.copyfile(source, target)
    protocol = {"version": 1, "configs": [config(seed) for seed in SEEDS], "source": provenance(),
                "split_sha256": digest(target), "test_previously_examined": True,
                "selection": "validation only; float roundtrip MAE within 1e-6 tied, then target forward+reverse MAE",
                "examples": "seed 17, first 25 test images in manifest order"}
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text())["protocol"] != protocol:
        raise ValueError("Protocol/source changed; use a new campaign directory")
    if not path.exists():
        save_json(path, {"created_utc": utc_now(), "protocol": protocol})
        # Keep exact experiment source alongside the protocol for reproducibility.
        archive = output / "source"
        archive.mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__), archive / "paper_core.py")
        shutil.copytree(ROOT / "alshicrypt", archive / "alshicrypt", ignore=shutil.ignore_patterns("__pycache__"))
    return manifest


def correlation(x, y):
    x = x.double().flatten(1)
    y = y.double().flatten(1)
    x = x - x.mean(1, keepdim=True)
    y = y - y.mean(1, keepdim=True)
    return ((x * y).sum(1) / (x.square().sum(1) * y.square().sum(1)).sqrt().clamp_min(1e-30)).tolist()


@torch.no_grad()
def evaluate(model, images, ids, schedule, device, split, domain, per_image=False):
    model.eval()
    generator = torch.Generator().manual_seed(seed_for("evaluation-noise", split, domain))
    records = {key: {} for key in ("forward", "reverse", "roundtrip", "analytic")}
    correlations = {"payload_rgb": [], "preview_rgb": []}
    exact = []
    payload_bytes = 0
    neural_seconds = 0.
    for start in range(0, len(images), 4):
        raw = images[start:start + 4]
        ref = raw.double() / 255
        n64 = schedule.noise(tuple(raw.shape), generator)
        x, noise, target = ref.float().to(device), n64.float().to(device), schedule.forward(ref, n64).float().to(device)
        synchronize(device)
        tick = time.perf_counter()
        encoded = model.encode(x, noise)
        reverse = model.decode(target, noise)
        synchronize(device)
        neural_seconds += time.perf_counter() - tick
        transmitted, size = serialize_roundtrip(encoded, "gaussian")
        payload_bytes += size
        synchronize(device)
        tick = time.perf_counter()
        restored = model.decode(transmitted, noise)
        synchronize(device)
        neural_seconds += time.perf_counter() - tick
        transmitted_target, _ = serialize_roundtrip(target, "gaussian")
        analytic = schedule.inverse(transmitted_target, noise)
        for name, prediction, truth in (("forward", encoded, target), ("reverse", reverse, x),
                                        ("roundtrip", restored, x), ("analytic", analytic, x)):
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"Nonfinite {name} output")
            for key, values in metrics(prediction, truth, raw[:, 3:].to(device), 1).items():
                records[name].setdefault(key, []).extend(values)
        rounded = (restored * 255).round().clamp(0, 255).to(torch.uint8)
        exact.extend(rounded.eq(raw.to(device)).flatten(1).all(1).double().tolist())
        correlations["payload_rgb"].extend(correlation(x[:, :3], encoded[:, :3]))
        preview = (encoded[:, :3].clamp(0, 1) * 255).round() / 255
        correlations["preview_rgb"].extend(correlation(x[:, :3], preview))
    result = {key: summarize(value) for key, value in records.items()}
    result.update({"count": len(images), "exact_rgba_image_rate": float(np.mean(exact)),
                   "payload_bytes_per_image": payload_bytes / len(images),
                   "noise_field_bytes_per_image": images[0].numel() * 4,
                   "neural_seconds_per_image_three_calls": neural_seconds / len(images),
                   "mean_correlation": {k: float(np.mean(v)) for k, v in correlations.items()}})
    if per_image:
        result["per_image"] = {"ids": ids, "metrics": records, "exact_rgba": exact, "correlations": correlations}
    return result


def evaluate_domains(model, pokemon, ids, schedule, device, split, per_image=False):
    static = synthetic_set(split)
    structured, probe_ids = probes()
    return {"pokemon": evaluate(model, pokemon, ids, schedule, device, split, "pokemon", per_image),
            "static": evaluate(model, static, [f"static_{i}" for i in range(len(static))], schedule, device, split, "static", per_image),
            "probes": evaluate(model, structured, probe_ids, schedule, device, split, "probes", per_image)}


def optimizer_step(model, optimizer, pokemon, cfg, data_rng, noise_rng, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = torch.zeros(2, device=device)
    schedule = Schedule(**cfg["schedule"])
    micro, batch = cfg["microbatch"], cfg["batch_size"]
    if micro < 1 or batch % micro:
        raise ValueError("Microbatch must divide batch")
    accumulation = batch // micro
    for _ in range(accumulation):
        raw = training_batch(pokemon, "pokemon", micro, data_rng)
        x = raw / 255  # Preserve the original study's training normalization.
        n64 = schedule.noise(tuple(raw.shape), noise_rng)
        target = schedule.forward(x.double(), n64)
        a, b = model.losses(x.to(device), target.float().to(device), n64.float().to(device))
        ((a + b) / accumulation).backward()
        losses += torch.stack([a.detach(), b.detach()]) / accumulation
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient")
    optimizer.step()
    return losses.tolist()


def train_one(cfg, manifest, output, data):
    folder = output / f"seed-{cfg['seed']}"
    folder.mkdir(parents=True, exist_ok=True)
    source = provenance()
    completed = folder / "result.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        if result["config"] != cfg or result["source"] != source:
            raise ValueError("Completed run differs from requested configuration/source")
        return result
    device = setup(cfg["seed"])
    schedule = Schedule(**cfg["schedule"])
    model = PaperCoupling(schedule, cfg["width"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    pokemon, _ = load_split(manifest, data, "train")
    val, ids = load_split(manifest, data, "val")
    dg = torch.Generator().manual_seed(seed_for("training-data", cfg["seed"]))
    ng = torch.Generator().manual_seed(seed_for("training-noise", cfg["seed"]))
    history, best_score, start, training_seconds, peak = [], None, 0, 0., 0
    last = folder / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["config"] != cfg or state["source"] != source:
            raise ValueError("Resume requires matching configuration and source")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        dg.set_state(state["data_rng"])
        ng.set_state(state["noise_rng"])
        history, best_score, start = state["history"], tuple(state["best_score"]), state["step"]
        untrained, training_seconds, peak = state["untrained"], state["training_seconds"], state["peak_allocated_bytes"]
    else:
        untrained = evaluate_domains(model, val, ids, schedule, device, "val")
    save_json(folder / "environment.json", environment())
    save_json(folder / "config.json", cfg)
    torch.cuda.reset_peak_memory_stats()
    interval_losses = np.zeros(2)
    interval_count = 0
    began = time.perf_counter()
    for step in range(start + 1, cfg["updates"] + 1):
        synchronize(device)
        tick = time.perf_counter()
        losses = optimizer_step(model, optimizer, pokemon, cfg, dg, ng, device)
        synchronize(device)
        training_seconds += time.perf_counter() - tick
        interval_losses += losses
        interval_count += 1
        if step % cfg["validation_interval"] == 0 or step == cfg["updates"]:
            if provenance() != source:
                raise ValueError("Experiment source changed during training")
            validation = evaluate_domains(model, val, ids, schedule, device, "val")
            score = selection_score(validation, "gaussian")
            history.append({"step": step, "mean_forward_loss": float(interval_losses[0] / interval_count),
                            "mean_reverse_loss": float(interval_losses[1] / interval_count), "validation": validation})
            interval_losses[:] = 0
            interval_count = 0
            if best_score is None or score < best_score:
                best_score = score
                atomic_checkpoint(folder / "best.pt", {"model": model.state_dict(), "config": cfg, "step": step, "source": source})
                save_json(folder / "best_validation.json", validation)
            peak = max(peak, torch.cuda.max_memory_allocated())
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg, "source": source,
                     "step": step, "data_rng": dg.get_state(), "noise_rng": ng.get_state(), "history": history,
                     "best_score": best_score, "untrained": untrained, "training_seconds": training_seconds,
                     "peak_allocated_bytes": peak}
            atomic_checkpoint(last, state)
            save_json(folder / "history.json", history)
            print(json.dumps({"seed": cfg["seed"], "update": step, "loss": history[-1]["mean_forward_loss"] + history[-1]["mean_reverse_loss"],
                              "validation_score": score}), flush=True)
    result = {"config": cfg, "source": source, "parameters": sum(p.numel() for p in model.parameters()),
              "validation": json.loads((folder / "best_validation.json").read_text()), "untrained": untrained,
              "training_seconds": training_seconds, "wall_seconds_this_session": time.perf_counter() - began,
              "peak_allocated_bytes": peak, "completed_utc": utc_now()}
    save_json(completed, result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def train(output=DEFAULT_RUN, data=DEFAULT_DATA):
    manifest = prepare(output, data)
    for seed in SEEDS:
        train_one(config(seed), manifest, output, data)
    selection = {"source": provenance(), "split_sha256": digest(output / "split.json"),
                 "protocol_sha256": digest(output / "protocol.json"), "test_previously_examined": True,
                 "candidates": [{"seed": seed, "run": f"seed-{seed}", "checkpoint_sha256": digest(output / f"seed-{seed}/best.pt")} for seed in SEEDS]}
    dest = output / "selection.json"
    if dest.exists() and json.loads(dest.read_text())["selection"] != selection:
        raise ValueError("Frozen selection differs")
    if not dest.exists():
        save_json(dest, {"frozen_utc": utc_now(), "selection": selection})


def verify_selection(output):
    selected = json.loads((output / "selection.json").read_text())["selection"]
    if selected["source"] != provenance():
        raise ValueError("Source changed after selection")
    if selected["split_sha256"] != digest(output / "split.json"):
        raise ValueError("Split changed after selection")
    if selected["protocol_sha256"] != digest(output / "protocol.json"):
        raise ValueError("Protocol changed after selection")
    for row in selected["candidates"]:
        if row["checkpoint_sha256"] != digest(output / row["run"] / "best.pt"):
            raise ValueError("Checkpoint changed after selection")
    return selected


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    schedule = Schedule(**state["config"]["schedule"])
    model = PaperCoupling(schedule, state["config"]["width"]).to(device).eval()
    model.load_state_dict(state["model"])
    return model, schedule, state


def test(output=DEFAULT_RUN, data=DEFAULT_DATA):
    selected = verify_selection(output)
    selection_hash = digest(output / "selection.json")
    marker = output / "test_opened.json"
    if marker.exists() and json.loads(marker.read_text())["selection_sha256"] != selection_hash:
        raise ValueError("Test selection changed")
    if not marker.exists():
        save_json(marker, {"opened_utc": utc_now(), "selection_sha256": selection_hash, "prior_study_test_access": True})
    manifest = json.loads((output / "split.json").read_text())
    validate_manifest(manifest, data)
    pokemon, ids = load_split(manifest, data, "test")
    device = setup(17)
    for row in selected["candidates"]:
        folder = output / row["run"]
        if (folder / "test.json").exists():
            if json.loads((folder / "test.json").read_text())["selection_sha256"] != selection_hash:
                raise ValueError("Existing test results have different provenance")
            continue
        model, schedule, _ = load_model(folder / "best.pt", device)
        results = evaluate_domains(model, pokemon, ids, schedule, device, "test", per_image=True)
        save_json(folder / "test.json", {"selection_sha256": selection_hash, "results": results})
        print(json.dumps({"test_seed": row["seed"], "pokemon_roundtrip": results["pokemon"]["roundtrip"],
                          "exact_rgba_image_rate": results["pokemon"]["exact_rgba_image_rate"]}), flush=True)
        del model
        torch.cuda.empty_cache()
