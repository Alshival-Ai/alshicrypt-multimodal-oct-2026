from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from .data import digest, load_split, make_manifest, save_json
from .evaluate import evaluate_domains, selection_score
from .models import make_model
from .train import environment, profile, run, setup, source_hash

TRACKS = ("byte", "gaussian")
FAMILIES = ("pair", "coupling")
REGIMES = ("pokemon", "static", "mixed")


def families_for(track):
    return (*FAMILIES, "guided") if track == "byte" else FAMILIES


def config(track, family, regime, seed=17, steps=1000, noise_mode="fresh", microbatch=4):
    return dict(track=track, family=family, regime=regime, seed=seed, steps=steps, noise_mode=noise_mode, microbatch=microbatch, batch_size=16, width=32, lr=3e-4)


def run_name(c):
    return f"{c['track']}_{c['family']}_{c['regime']}_{c['noise_mode']}_s{c['seed']}"


def pilot_campaign(root: Path, output: Path, max_steps=1000, budget_seconds=1800, profile_steps=100):
    if max_steps > 1000 or max_steps < 1:
        raise ValueError("Screening must be 1..1000 updates (<=10 epochs)")
    output.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(root, output / "split.json")
    train, _ = load_split(manifest, root, "train")
    profiles = {}
    profile_path = output / "profiles.json"
    if profile_path.exists():
        profiles = json.loads(profile_path.read_text())
    for track, family in [(t, f) for t in TRACKS for f in families_for(t)]:
        key = f"{track}_{family}"
        if key in profiles:
            continue
        c = config(track, family, "mixed")
        while True:
            try:
                result = profile(c, train, profile_steps)
                break
            except torch.cuda.OutOfMemoryError:
                if c["microbatch"] <= 1:
                    raise
                c["microbatch"] //= 2
                torch.cuda.empty_cache()
        profiles[key] = {**result, "microbatch": c["microbatch"]}
        save_json(profile_path, profiles)
        print(json.dumps({"profile": key, **profiles[key]}), flush=True)
    slowest = max(v["seconds_per_step"] for v in profiles.values())
    steps = max(1, min(max_steps, math.floor(budget_seconds * .8 / slowest)))
    if steps >= 100:
        steps = steps // 100 * 100
    schedule = {"steps": steps, "max_steps": max_steps, "seconds_budget": budget_seconds, "profile_steps": profile_steps, "environment": environment(), "split_sha256": digest(output / "split.json")}
    schedule_path = output / "screening_schedule.json"
    if schedule_path.exists():
        old = json.loads(schedule_path.read_text())
        if any(old[k] != schedule[k] for k in ["steps", "max_steps", "seconds_budget", "split_sha256"]):
            raise ValueError("Use a new campaign directory for a different schedule")
    else:
        save_json(schedule_path, schedule)
    results = []
    for track, family, regime in [(t, f, r) for t in TRACKS for f in families_for(t) for r in REGIMES]:
        c = config(track, family, regime, steps=steps, microbatch=profiles[f"{track}_{family}"]["microbatch"])
        results.append(run(c, manifest, root, output / "screening" / run_name(c)))
    promoted = {}
    for track in TRACKS:
        candidates = [r for r in results if r["config"]["track"] == track]
        # Average over every data regime, not the most favorable single run.
        scores = {}
        for family in families_for(track):
            rows = [r for r in candidates if r["config"]["family"] == family]
            scores[family] = tuple(np.mean([selection_score(r["validation"], track) for r in rows], axis=0))
        promoted[track] = min(families_for(track), key=lambda family: scores[family])
    save_json(output / "promotion.json", {"families": promoted, "screening_steps": steps, "selection": "mean validation score across all three training regimes"})
    return promoted


def extended_campaign(root: Path, output: Path, max_steps=10000, budget_seconds=7200):
    if not 1 <= max_steps <= 10000:
        raise ValueError("Extended trials must be <=100 epochs")
    promotion = json.loads((output / "promotion.json").read_text())
    manifest = make_manifest(root, output / "split.json")
    profiles = json.loads((output / "profiles.json").read_text())
    slowest = max(profiles[f"{t}_{f}"]["seconds_per_step"] for t, f in promotion["families"].items())
    steps = min(max_steps, max(1, int(budget_seconds * .8 / slowest)))
    if steps >= 100:
        steps = steps // 100 * 100
    results = []
    for track, regime, seed in itertools.product(TRACKS, REGIMES, [17, 29, 43]):
        family = promotion["families"][track]
        c = config(track, family, regime, seed, steps, microbatch=profiles[f"{track}_{family}"]["microbatch"])
        result = run(c, manifest, root, output / "extended" / run_name(c))
        results.append((c, result))
    # Fixed-noise diagnostic uses the same update budget as its fresh counterpart.
    for track in TRACKS:
        family = promotion["families"][track]
        c = config(track, family, "mixed", 17, steps, "fixed", profiles[f"{track}_{family}"]["microbatch"])
        run(c, manifest, root, output / "diagnostics" / run_name(c))
    frozen = []
    winners = {}
    for track in TRACKS:
        def regime_score(regime):
            rows = [r for c, r in results if c["track"] == track and c["regime"] == regime]
            return tuple(np.mean([selection_score(r["validation"], track) for r in rows], axis=0))
        winners[track] = min(REGIMES, key=regime_score)
    for c, result in results:
        folder = output / "extended" / run_name(c)
        frozen.append({"run": str(folder.relative_to(output)), "config": c, "checkpoint_sha256": digest(folder / "best.pt"), "validation": result["validation"]})
    evidence = {}
    for track in TRACKS:
        rows = [r for c, r in results if c["track"] == track and c["regime"] == winners[track]]
        before = np.mean([r["untrained"]["pokemon"]["forward"]["mae"] + r["untrained"]["pokemon"]["reverse"]["mae"] for r in rows])
        after = np.mean([r["validation"]["pokemon"]["forward"]["mae"] + r["validation"]["pokemon"]["reverse"]["mae"] for r in rows])
        evidence[track] = {"untrained_target_mae_sum": float(before), "trained_target_mae_sum": float(after), "relative_improvement": float(1 - after / max(before, 1e-12))}
    primary = max(TRACKS, key=lambda t: evidence[t]["relative_improvement"])
    selection = {"version": 1, "families": promotion["families"], "winning_regimes": winners, "split_sha256": digest(output / "split.json"), "source_sha256": source_hash(), "candidates": frozen, "test_used_for_selection": False, "primary_track": primary, "primary_track_reason": "Largest validation target-learning improvement relative to its untrained control; absolute errors and transport differences must also be reported.", "learning_evidence": evidence, "gaussian_tolerance": 1e-6}
    destination = output / "selection.json"
    if destination.exists() and json.loads(destination.read_text()) != selection:
        raise ValueError("Selection already frozen; create a new campaign")
    save_json(destination, selection)
    return selection


def sealed_test(root: Path, output: Path):
    selection_path = output / "selection.json"
    selected = json.loads(selection_path.read_text())
    selection_hash = digest(selection_path)
    marker = output / "test_opened.json"
    if marker.exists() and json.loads(marker.read_text())["selection_sha256"] != selection_hash:
        raise ValueError("Test already opened under another selection")
    if selected["split_sha256"] != digest(output / "split.json"):
        raise ValueError("Split changed after selection")
    for entry in selected["candidates"]:
        if digest(output / entry["run"] / "best.pt") != entry["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after selection")
    if selected["source_sha256"] != source_hash():
        raise ValueError("Experiment source changed after selection")
    # Mark access BEFORE loading test pixels. Partial evaluations resume safely.
    save_json(marker, {"selection_sha256": selection_hash, "purpose": "one frozen evaluation; results must not tune this campaign"})
    manifest = make_manifest(root, output / "split.json")
    test, ids = load_split(manifest, root, "test")
    device = setup(17)
    for entry in selected["candidates"]:
        folder = output / entry["run"]
        dest = folder / "test.json"
        if dest.exists():
            continue
        c = entry["config"]
        model = make_model(c["family"], c["track"], c["width"]).to(device)
        state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        result = evaluate_domains(model, test, ids, c["track"], device, "test", per_image=True)
        save_json(dest, {"selection_sha256": selection_hash, "results": result})
        print(json.dumps({"test": entry["run"], "roundtrip": result["pokemon"]["roundtrip"]}), flush=True)
        del model
        torch.cuda.empty_cache()
