import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from alshicrypt import processes as p
from alshicrypt.data import make_manifest, seed_for, synthetic_set, validate_manifest
from alshicrypt.evaluate import metrics, selection_score
from alshicrypt.models import make_model


@pytest.mark.parametrize("track", ["byte", "gaussian"])
def test_analytic_inverse_and_serialization(track):
    g = torch.Generator().manual_seed(1)
    x = torch.randint(256, (3, 4, 9, 11), generator=g).double()
    if track == "gaussian":
        x /= 255
    n = p.noise_field(tuple(x.shape), track, g)
    y = p.forward(x, n, track)
    assert torch.allclose(p.inverse(y, n, track), x, atol=1e-14, rtol=0)
    transmitted, count = p.serialize_roundtrip(y, track)
    assert count > 0
    restored = p.inverse(transmitted, n.float(), track)
    assert torch.allclose(restored.double(), x, atol=1e-6, rtol=0)


def test_byte_boundaries_exhaustive():
    x = torch.arange(256)[:, None]
    n = torch.arange(256)[None, :]
    assert torch.equal(p.inverse(p.forward(x, n, "byte"), n, "byte"), x.expand(256, 256))


@pytest.mark.parametrize("track", ["byte", "gaussian"])
def test_coupling_roundtrip_reload_and_gradient(track):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = make_model("coupling", track, 8)
    x = torch.randint(256, (2, 4, 8, 8)).float()
    if track == "gaussian":
        x /= 255
    n = p.noise_field(tuple(x.shape), track, torch.Generator().manual_seed(3)).float()
    y = model.encode(x, n)
    restored = model.decode(y, n)
    assert torch.allclose(restored, x, atol=0 if track == "byte" else 1e-6, rtol=0)
    a, b = model.losses(x, p.forward(x, n, track), n)
    (a + b).backward()
    assert any(t.grad is not None and t.grad.abs().sum() > 0 for t in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    copy = make_model("coupling", track, 8)
    copy.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(y, copy.encode(x, n))


def test_split_groups_and_tamper_detection(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    g = np.random.default_rng(4)
    for i in range(30):
        Image.fromarray(g.integers(0, 256, (120, 120, 4), dtype=np.uint8)).save(root / f"pokemon{i}.png")
    (root / "pokemon0-alt.png").write_bytes((root / "pokemon0.png").read_bytes())
    m = make_manifest(root, tmp_path / "split.json")
    validate_manifest(m, root)
    assert len({r["split"] for r in m["images"] if r["id"].startswith("pokemon0")}) == 1
    assert set(r["split"] for r in m["images"]) == {"train", "val", "test"}
    assert m == make_manifest(root, tmp_path / "split.json")
    (root / "pokemon0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Dataset changed"):
        validate_manifest(m, root)


def test_static_streams_and_metric_exactness():
    assert torch.equal(synthetic_set("val", 2, 8), synthetic_set("val", 2, 8))
    assert not torch.equal(synthetic_set("val", 2, 8), synthetic_set("test", 2, 8))
    x = torch.zeros(1, 4, 2, 2)
    y = x.clone()
    y[0, 0, 0, 0] = 1
    result = metrics(y, x, torch.ones(1, 1, 2, 2), 255)
    assert result["exact_image_rate"] == [0.]
    assert result["byte_accuracy"] == [15 / 16]


def test_quantization_is_not_inversion():
    x = torch.tensor([.1, .9])
    n = torch.tensor([2., 2.])
    y = p.forward(x, n, "gaussian")
    restored = p.inverse(y.clamp(0, 1).mul(255).round().div(255), n, "gaussian")
    assert not torch.allclose(x, restored)


def test_gaussian_selection_treats_roundoff_as_tie():
    def row(error, target):
        return {"pokemon": {"roundtrip": {"mae": error}, "forward": {"mae": target}, "reverse": {"mae": target}}}
    assert selection_score(row(1e-7, .1), "gaussian") < selection_score(row(1e-8, .2), "gaussian")


def test_sealed_test_rejects_changed_selection_before_loading(tmp_path, monkeypatch):
    from alshicrypt.campaign import sealed_test
    from alshicrypt.data import save_json
    save_json(tmp_path / "selection.json", {"anything": "changed"})
    save_json(tmp_path / "test_opened.json", {"selection_sha256": "previous"})
    monkeypatch.setattr("alshicrypt.campaign.load_split", lambda *a: pytest.fail("Test pixels must not be loaded"))
    with pytest.raises(ValueError, match="already opened"):
        sealed_test(tmp_path, tmp_path)


def test_sealed_test_rejects_changed_checkpoint(tmp_path, monkeypatch):
    from alshicrypt.campaign import sealed_test
    from alshicrypt.data import digest, save_json
    save_json(tmp_path / "split.json", {})
    run = tmp_path / "extended" / "run"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"changed")
    save_json(tmp_path / "selection.json", {"split_sha256": digest(tmp_path / "split.json"), "candidates": [{"run": "extended/run", "checkpoint_sha256": "old"}]})
    monkeypatch.setattr("alshicrypt.campaign.load_split", lambda *a: pytest.fail("Test pixels must not be loaded"))
    with pytest.raises(ValueError, match="Checkpoint changed"):
        sealed_test(tmp_path, tmp_path)
    assert not (tmp_path / "test_opened.json").exists()


def test_gaussian_endpoint_matches_iterative_process():
    g = torch.Generator().manual_seed(71)
    x = torch.rand(2, 4, 8, 8, generator=g, dtype=torch.float64)
    y, noise = x.clone(), torch.zeros_like(x)
    a, s = (1 - p.BETA)**.5, p.BETA**.5
    for _ in range(p.STEPS):
        eps = torch.randn(x.shape, generator=g, dtype=x.dtype)
        y = a * y + s * eps
        noise = a * noise + s * eps
    assert torch.allclose(y, p.forward(x, noise, "gaussian"), rtol=0, atol=1e-14)


def test_guidance_preserves_architecture_and_has_gradients():
    torch.manual_seed(5)
    a = make_model("coupling", "byte", 8)
    b = make_model("guided", "byte", 8)
    b.load_state_dict(a.state_dict())
    x = torch.randint(256, (2, 4, 8, 8)).float()
    n = torch.randint(256, x.shape).float()
    assert torch.equal(a.encode(x, n), b.encode(x, n))
    assert torch.equal(b.decode(b.encode(x, n), n), x)
    losses = b.losses(x, p.forward(x, n, "byte"), n)
    sum(losses).backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in b.parameters())


def test_microbatch_one_preserves_mixed_ratio(monkeypatch):
    from alshicrypt.train import optimizer_step
    from alshicrypt.data import training_batch
    calls = []
    def spy(pokemon, regime, batch, generator):
        calls.append(regime)
        return training_batch(pokemon, regime, batch, generator)
    monkeypatch.setattr("alshicrypt.train.training_batch", spy)
    model = make_model("coupling", "gaussian", 4)
    optimizer = torch.optim.AdamW(model.parameters())
    config = {"track": "gaussian", "family": "coupling", "regime": "mixed", "microbatch": 1, "batch_size": 4}
    pokemon = torch.zeros(2, 4, 4, 4, dtype=torch.uint8)
    optimizer_step(model, optimizer, pokemon, config, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2), torch.device("cpu"))
    assert calls == ["pokemon", "static", "pokemon", "static"]


def test_report_does_not_reuse_old_conclusions(tmp_path, monkeypatch):
    from alshicrypt.data import save_json
    from alshicrypt.report import report
    monkeypatch.chdir(tmp_path)
    old = tmp_path / "paper" / "generated"
    old.mkdir(parents=True)
    (old / "conclusions.tex").write_text("stale conclusion")
    (old / "selection.json").write_text("stale selection")
    folder = tmp_path / "runs" / "new" / "screening" / "example"
    c = dict(track="byte", family="pair", regime="pokemon", seed=17, steps=1, noise_mode="fresh")
    metric = {"mae": .1}
    val = {"pokemon": {"forward": metric, "reverse": metric, "roundtrip": metric}}
    save_json(folder / "result.json", dict(config=c, validation=val, untrained=val, parameters=1, training_seconds=1, peak_allocated_bytes=1))
    save_json(folder / "environment.json", {})
    save_json(folder / "history.json", [])
    report(folder.parent.parent)
    assert not (old / "conclusions.tex").exists()
    assert not (old / "selection.json").exists()
    assert "validation" in (old / "status.tex").read_text()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("track", ["byte", "gaussian"])
def test_gpu_coupling_serialized_roundtrip(track):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(13)
    model = make_model("coupling", track, 8).cuda().eval()
    x = torch.randint(256, (3, 4, 16, 16), device="cuda").float()
    if track == "gaussian":
        x /= 255
    noise = p.noise_field(tuple(x.shape), track, torch.Generator().manual_seed(71)).float().cuda()
    with torch.no_grad():
        y, _ = p.serialize_roundtrip(model.encode(x, noise), track)
        restored = model.decode(y, noise)
    assert torch.allclose(x, restored, rtol=0, atol=0 if track == "byte" else 1e-6)
