"""Package only the final manuscript's LaTeX source and referenced assets."""
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def main():
    paper = Path(__file__).resolve().parents[1] / "paper"
    pending = [Path("main.tex")]
    included = set(pending)
    while pending:
        source = pending.pop()
        references = re.findall(
            r"\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}",
            (paper / source).read_text(),
        )
        for name in references:
            asset = Path(name)
            if not asset.suffix:
                asset = asset.with_suffix(".tex")
            if asset.is_absolute() or ".." in asset.parts:
                raise ValueError(f"Asset must be inside the manuscript directory: {asset}")
            if not (paper / asset).is_file():
                raise FileNotFoundError(paper / asset)
            if asset not in included:
                included.add(asset)
                if asset.suffix == ".tex":
                    pending.append(asset)
    destination = paper / "arxiv-source.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for asset in sorted(included):
            entry = ZipInfo(asset.as_posix(), date_time=(2026, 10, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, (paper / asset).read_bytes())
    print(f"Packaged {len(included)} files in {destination}")


if __name__ == "__main__":
    main()
