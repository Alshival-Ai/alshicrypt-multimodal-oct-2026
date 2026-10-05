from __future__ import annotations

import contextlib
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from . import processes as p
from .data import load_split, save_json, seed_for, training_batch
from .evaluate import evaluate_domains, selection_score, synchronize
from .models import make_model


def source_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def setup(seed: int, require_cuda: bool = True) -> torch.device:
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for campaign; CPU permitted only for tests")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def environment() -> dict:
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "source_sha256": source_hash(), "deterministic_cudnn": True, "tf32": False}


def inputs(raw, config, g, device):
    x = raw if config["track"] == "byte" else raw / 255
    if config.get("noise_mode", "fresh") == "fixed":
        g = torch.Generator().manual_seed(seed_for("fixed-noise", 17))
        noise = p.noise_field(tuple(x.shape), config["track"], g, fixed=True).expand_as(x)
    else:
        noise = p.noise_field(tuple(x.shape), config["track"], g)
    y = p.forward(x.double(), noise, config["track"])
    return x.to(device), y.float().to(device), noise.float().to(device)


def optimizer_step(model, optimizer, pokemon, config, data_g, noise_g, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = config["microbatch"]
    if micro < 1 or config["batch_size"] % micro:
        raise ValueError("Microbatch must divide effective batch size")
    accumulated = config["batch_size"] // micro
    total = torch.zeros(2, device=device)
    for part in range(accumulated):
        regime = config["regime"]
        if regime == "mixed" and micro == 1:
            if config["batch_size"] % 2:
                raise ValueError("Mixed training requires an even effective batch")
            regime = "pokemon" if part % 2 == 0 else "static"
        raw = training_batch(pokemon, regime, micro, data_g)
        x, y, n = inputs(raw, config, noise_g, device)
        # Rounding-dependent discrete couplings must use consistent FP32 arithmetic.
        amp = config["family"] == "pair" and device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext():
            a, b = model.losses(x, y, n)
            loss = (a + b) / accumulated
        loss.backward()
        total += torch.stack([a.detach(), b.detach()]).float() / accumulated
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient")
    optimizer.step()
    return total.tolist()


def profile(config: dict, pokemon: torch.Tensor, steps: int = 100) -> dict:
    device = setup(config["seed"])
    model = make_model(config["family"], config["track"], config["width"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    dg = torch.Generator().manual_seed(1701)
    ng = torch.Generator().manual_seed(1702)
    for _ in range(3):
        optimizer_step(model, optimizer, pokemon, config, dg, ng, device)
    torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer_step(model, optimizer, pokemon, config, dg, ng, device)
    synchronize(device)
    result = {"seconds_per_step": (time.perf_counter() - start) / steps, "steps": steps, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "parameters": sum(p.numel() for p in model.parameters())}
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def run(config: dict, manifest: dict, root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        result = json.loads((output / "result.json").read_text())
        if result["config"] != config:
            raise ValueError(f"Refusing to overwrite a different run: {output}")
        return result
    if config["steps"] > 10000:
        raise ValueError("Short trial limit: 100 epochs of 100 updates")
    device = setup(config["seed"])
    pokemon, _ = load_split(manifest, root, "train")
    val, ids = load_split(manifest, root, "val")
    model = make_model(config["family"], config["track"], config["width"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    data_g = torch.Generator().manual_seed(seed_for("training-data", config["seed"]))
    noise_g = torch.Generator().manual_seed(seed_for("training-noise", config["seed"]))
    start_step = 0
    history = []
    best_score = None
    untrained = None
    training_seconds = 0.
    checkpoint_path = output / "last.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["config"] != config or state["source_sha256"] != source_hash():
            raise ValueError("Resume requires matching config and source")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        data_g.set_state(state["data_rng"])
        noise_g.set_state(state["noise_rng"])
        start_step, history = state["step"], state["history"]
        best_score, untrained, training_seconds = tuple(state["best_score"]), state["untrained"], state["training_seconds"]
    if untrained is None:
        untrained = evaluate_domains(model, val, ids, config["track"], device, "val", config["noise_mode"])
    save_json(output / "config.json", config)
    save_json(output / "environment.json", environment())
    torch.cuda.reset_peak_memory_stats()
    interval = min(100, config["steps"])
    begin = time.perf_counter()
    for step in range(start_step + 1, config["steps"] + 1):
        synchronize(device)
        tick = time.perf_counter()
        a, b = optimizer_step(model, optimizer, pokemon, config, data_g, noise_g, device)
        synchronize(device)
        training_seconds += time.perf_counter() - tick
        if step % interval == 0 or step == config["steps"]:
            result = evaluate_domains(model, val, ids, config["track"], device, "val", config["noise_mode"])
            score = selection_score(result, config["track"])
            history.append({"step": step, "epoch": step / 100, "forward_loss": a, "reverse_loss": b, "validation": result})
            if best_score is None or score < best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "config": config, "step": step}, output / "best.pt")
                save_json(output / "best_validation.json", result)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config, "step": step, "source_sha256": source_hash(), "data_rng": data_g.get_state(), "noise_rng": noise_g.get_state(), "history": history, "best_score": best_score, "untrained": untrained, "training_seconds": training_seconds}
            tmp = output / "last.tmp"
            torch.save(state, tmp)
            tmp.replace(checkpoint_path)
            save_json(output / "history.json", history)
            print(json.dumps({"run": output.name, "step": step, "loss": [a, b], "score": score}), flush=True)
    best = json.loads((output / "best_validation.json").read_text())
    result = {"config": config, "validation": best, "untrained": untrained, "parameters": sum(p.numel() for p in model.parameters()), "training_seconds": training_seconds, "wall_seconds_this_session": time.perf_counter() - begin, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "completed_steps": config["steps"]}
    save_json(output / "result.json", result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result
