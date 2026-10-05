from __future__ import annotations

import argparse
from pathlib import Path

from .data import DEFAULT_DATA, make_manifest


def main():
    parser = argparse.ArgumentParser(description="Reproducible stochastic image research")
    parser.add_argument("command", choices=["prepare", "screen", "extend", "test", "diagnose", "report", "all"])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("runs/campaign-v1"))
    parser.add_argument("--screen-steps", type=int, default=1000)
    parser.add_argument("--extended-steps", type=int, default=10000)
    parser.add_argument("--screen-seconds", type=int, default=1800)
    parser.add_argument("--extended-seconds", type=int, default=7200)
    parser.add_argument("--profile-steps", type=int, default=100)
    args = parser.parse_args()
    if args.command == "prepare":
        print(make_manifest(args.data, args.output / "split.json")["counts"])
        return
    from .campaign import extended_campaign, pilot_campaign, sealed_test
    if args.command in {"screen", "all"}:
        pilot_campaign(args.data, args.output, args.screen_steps, args.screen_seconds, args.profile_steps)
    if args.command in {"extend", "all"}:
        extended_campaign(args.data, args.output, args.extended_steps, args.extended_seconds)
    if args.command in {"test", "all"}:
        sealed_test(args.data, args.output)
    if args.command in {"diagnose", "all"}:
        from .diagnostics import diagnose
        diagnose(args.data, args.output)
    if args.command in {"report", "all"}:
        from .report import report
        report(args.output)


if __name__ == "__main__":
    main()
