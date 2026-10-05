from __future__ import annotations

import argparse
from pathlib import Path

from .core import DEFAULT_DATA, DEFAULT_RUN, train, test


def main():
    parser = argparse.ArgumentParser(description="32-step paper study: train, evaluate, render, and compile")
    parser.add_argument("command", choices=("train", "evaluate", "illustrate", "explain", "render", "build", "all"))
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--paper", type=Path, default=Path(__file__).resolve().parents[1] / "paper")
    args = parser.parse_args()
    if args.command == "illustrate":
        from .report import stochastic_figure, coupling_figure, process_figure, geometry_figure
        stochastic_figure(args.data, args.paper)
        coupling_figure(args.data, args.paper)
        process_figure(args.paper)
        geometry_figure(args.paper)
    if args.command == "explain":
        from .report import process_figure, geometry_figure, saved_result_assets
        process_figure(args.paper)
        geometry_figure(args.paper)
        saved_result_assets(args.output, args.paper)
    if args.command in ("train", "all"):
        train(args.output, args.data)
    if args.command in ("evaluate", "all"):
        test(args.output, args.data)
    if args.command in ("render", "all"):
        from .report import render
        render(args.output, args.data, args.paper)
    if args.command in ("build", "all"):
        from .report import build
        build(args.paper)


if __name__ == "__main__":
    main()
