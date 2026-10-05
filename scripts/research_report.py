"""Render the frozen legacy report into research/, without editing its source."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alshicrypt.report import report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/campaign-v1")
    args, _ = parser.parse_known_args()
    campaign = args.output.resolve()
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="alshicrypt-research-report-") as stage:
        try:
            os.chdir(stage)
            report(campaign)
        finally:
            os.chdir(previous)
        shutil.copytree(Path(stage) / "paper/generated", ROOT / "research/generated", dirs_exist_ok=True)
