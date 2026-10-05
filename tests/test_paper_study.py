import json

import pytest
import torch

from alshicrypt.data import digest, save_json
from alshicrypt.processes import serialize_roundtrip
from paper_study.core import Schedule, PaperCoupling, config, provenance, verify_selection, load_model, atomic_checkpoint


def test_32_step_endpoint_and_inverse():
    s = Schedule()
    assert s.signal == pytest.approx(.25)
    assert s.noise_sd ** 2 == pytest.approx(.9375)
    rng = torch.Generator().manual_seed(8)
    x = torch.rand(2, 4, 9, 9, dtype=torch.float64, generator=rng)
    y, n = x.clone(), torch.zeros_like(x)
    for _ in range(s.steps):
        eps = torch.randn(x.shape, dtype=x.dtype, generator=rng)
        y = (1 - s.beta) ** .5 * y + s.beta ** .5 * eps
        n = (1 - s.beta) ** .5 * n + s.beta ** .5 * eps
    assert torch.allclose(y, s.forward(x, n), atol=1e-14, rtol=0)
    assert torch.allclose(s.inverse(y, n), x, atol=1e-14, rtol=0)


def test_schedule_is_explicit_and_changes_noise():
    a, b = Schedule(16), Schedule(32)
    assert a.signal == pytest.approx(.5)
    n1 = a.noise((2, 4, 9, 9), torch.Generator().manual_seed(7))
    n2 = b.noise((2, 4, 9, 9), torch.Generator().manual_seed(7))
    assert torch.allclose(n1 / a.noise_sd, n2 / b.noise_sd)
    with pytest.raises(ValueError):
        Schedule(0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_paper_serialization_reload_and_gradient(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(3)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = config(17)
    model = PaperCoupling().to(device)
    assert sum(p.numel() for p in model.parameters()) == 44296
    assert model.scale == pytest.approx(.5)
    raw = torch.randint(256, (2, 4, 12, 12), device=device)
    x = raw.float() / 255
    n = Schedule().noise(x.shape, torch.Generator().manual_seed(2)).float().to(device)
    y = model.encode(x, n)
    payload, _ = serialize_roundtrip(y, "gaussian")
    restored = model.decode(payload, n)
    assert torch.equal((restored * 255).round().long(), raw)
    path = tmp_path / "model.pt"
    atomic_checkpoint(path, {"config": cfg, "model": model.state_dict()})
    copy, schedule, _ = load_model(path, device)
    assert schedule.steps == 32
    assert torch.equal(copy.encode(x, n), y)
    sum(model.losses(x, Schedule().forward(x, n), n)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_frozen_selection_checks_all_provenance(tmp_path):
    save_json(tmp_path / "split.json", {})
    save_json(tmp_path / "protocol.json", {})
    run = tmp_path / "seed-17"
    run.mkdir()
    (run / "best.pt").write_bytes(b"unchanged")
    selected = {"source": provenance(), "split_sha256": digest(tmp_path / "split.json"),
                "protocol_sha256": digest(tmp_path / "protocol.json"),
                "candidates": [{"run": "seed-17", "checkpoint_sha256": digest(run / "best.pt")}]}
    save_json(tmp_path / "selection.json", {"selection": selected})
    assert verify_selection(tmp_path) == selected
    (run / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        verify_selection(tmp_path)
    (run / "best.pt").write_bytes(b"unchanged")
    selected["source"]["paper_core_sha256"] = "wrong"
    save_json(tmp_path / "selection.json", {"selection": selected})
    with pytest.raises(ValueError, match="Source changed"):
        verify_selection(tmp_path)
