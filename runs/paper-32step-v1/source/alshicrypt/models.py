from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + .1 * self.net(x)


class ResidualMap(nn.Module):
    def __init__(self, track: str, width: int = 32, blocks: int = 6):
        super().__init__()
        self.track = track
        self.net = nn.Sequential(nn.Conv2d(8, width, 3, padding=1), nn.SiLU(), *[ResidualBlock(width) for _ in range(blocks)], nn.Conv2d(width, 1024 if track == "byte" else 4, 1))

    def forward(self, x, noise):
        scale = 255 if self.track == "byte" else 1
        result = self.net(torch.cat([x / scale, noise.expand_as(x) / scale], 1))
        return result.reshape(x.shape[0], 4, 256, *x.shape[-2:]) if self.track == "byte" else result


class Pair(nn.Module):
    def __init__(self, track: str, width: int = 32, blocks: int = 6):
        super().__init__()
        self.track = track
        self.encoder = ResidualMap(track, width, blocks)
        self.decoder = ResidualMap(track, width, blocks)

    def encode(self, x, noise):
        y = self.encoder(x, noise)
        return y.argmax(2).float() if self.track == "byte" else y

    def decode(self, y, noise):
        x = self.decoder(y, noise)
        return x.argmax(2).float() if self.track == "byte" else x

    def losses(self, x, y, noise):
        a, b = self.encoder(x, noise), self.decoder(y, noise)
        if self.track == "byte":
            return F.cross_entropy(a.flatten(0, 1), y.long().flatten(0, 1)), F.cross_entropy(b.flatten(0, 1), x.long().flatten(0, 1))
        return F.mse_loss(a, y), F.mse_loss(b, x)


class Coupling(nn.Module):
    def __init__(self, track: str, width: int = 32, blocks: int = 4):
        super().__init__()
        self.track = track
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(6, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 2, 1)) for _ in range(blocks)])
        # Each half is updated twice. This fixed scale matches total attenuation,
        # while the conditional additive functions must still be learned.
        self.scale = .5 ** (2 / blocks) if track == "gaussian" else 1.

    def transform(self, x, noise, reverse=False):
        n = noise.expand_as(x) / (255 if self.track == "byte" else 1)
        indices = range(len(self.layers) - 1, -1, -1) if reverse else range(len(self.layers))
        for i in indices:
            a, b = (x[:, :2], x[:, 2:]) if i % 2 == 0 else (x[:, 2:], x[:, :2])
            shift = self.layers[i](torch.cat([a / (255 if self.track == "byte" else 1), n], 1))
            if self.track == "byte":
                shift = shift * 128
                shift = shift + (shift.round() - shift).detach()
                b = torch.remainder(b - shift if reverse else b + shift, 256)
            else:
                b = (b - shift) / self.scale if reverse else self.scale * b + shift
            x = torch.cat([a, b], 1) if i % 2 == 0 else torch.cat([b, a], 1)
        return x

    def encode(self, x, noise):
        return self.transform(x, noise)

    def decode(self, y, noise):
        return self.transform(y, noise, reverse=True)

    def losses(self, x, y, noise):
        a, b = self.encode(x, noise), self.decode(y, noise)
        if self.track == "byte":
            return (1 - torch.cos((a - y) * (2 * math.pi / 256))).mean(), (1 - torch.cos((b - x) * (2 * math.pi / 256))).mean()
        return F.mse_loss(a, y), F.mse_loss(b, x)


class GuidedCoupling(Coupling):
    """Same invertible architecture, plus supervision of a teacher decomposition.

    Two integer updates per half sum to the cumulative field. This provides
    unwrapped shift targets and avoids relying solely on a periodic endpoint loss.
    The additional teacher information is declared as a training intervention.
    """
    def losses(self, x, y, noise):
        a, b = super().losses(x, y, noise)
        state = x
        extra = x.new_zeros(())
        n = noise.expand_as(x)
        for i, layer in enumerate(self.layers):
            left, right = (state[:, :2], state[:, 2:]) if i % 2 == 0 else (state[:, 2:], state[:, :2])
            raw_shift = layer(torch.cat([left / 255, n / 255], 1)) * 128
            half_noise = n[:, 2:] if i % 2 == 0 else n[:, :2]
            target = torch.floor(half_noise / 2) if i < 2 else half_noise - torch.floor(half_noise / 2)
            extra = extra + F.mse_loss(raw_shift / 128, target / 128) / len(self.layers)
            # Teacher forcing isolates the effect of exposing intermediate targets.
            right = torch.remainder(right + target, 256)
            state = torch.cat([left, right], 1) if i % 2 == 0 else torch.cat([right, left], 1)
        return a + extra, b + extra


def make_model(family: str, track: str, width: int = 32) -> nn.Module:
    if family not in {"pair", "coupling", "guided"} or track not in {"byte", "gaussian"}:
        raise ValueError((family, track))
    if family == "guided":
        if track != "byte":
            raise ValueError("Intermediate integer supervision is byte-specific")
        return GuidedCoupling(track, width)
    return (Pair if family == "pair" else Coupling)(track, width)
