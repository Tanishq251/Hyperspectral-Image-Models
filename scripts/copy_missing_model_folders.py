#!/usr/bin/env python3
"""Copy missing FGKAN folders into matching dataset folders.

Expected structure:
- Source:      Results/FGKAN/<dataset>
- Destination: Results/Samples_30_10_(PCA30)_(32)_(11)/<dataset>/FGKAN

For each dataset folder in the source, the script checks whether the same
 dataset exists in the destination. If it does, and the destination FGKAN
 folder is missing, the source dataset folder is copied into it.

Nothing existing in the destination is overwritten.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy Results/FGKAN/<dataset> into "
            "Results/Samples_30_10_(PCA30)_(32)_(11)/<dataset>/FGKAN when missing."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/aryan/Tanishq/Hyperspectral_library/Results/FGKAN"),
        help="Source directory containing dataset folders.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(
            "/home/aryan/Tanishq/Hyperspectral_library/Results/Samples_30_10_(PCA30)_(32)_(11)"
        ),
        help="Destination directory containing dataset folders.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()

    if not source.is_dir():
        print(f"Error: source directory does not exist or is not a directory: {source}", file=sys.stderr)
        return 1

    destination.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped: list[str] = []
    missing_target_datasets: list[str] = []

    for dataset_dir in sorted(source.iterdir(), key=lambda p: p.name.lower()):
        if not dataset_dir.is_dir():
            continue

        target_dataset = destination / dataset_dir.name
        if not target_dataset.is_dir():
            missing_target_datasets.append(dataset_dir.name)
            continue

        target_fgkan = target_dataset / "FGKAN"
        if target_fgkan.exists():
            skipped.append(dataset_dir.name)
            continue

        shutil.copytree(dataset_dir, target_fgkan, copy_function=shutil.copy2, symlinks=True)
        copied.append(dataset_dir.name)

    print("Summary")
    print("-------")
    print(f"Source: {source}")
    print(f"Destination: {destination}")
    print(f"Datasets copied: {len(copied)}")
    for name in copied:
        print(f"  - {name}")
    print(f"Datasets skipped (FGKAN already present): {len(skipped)}")
    for name in skipped:
        print(f"  - {name}")
    print(f"Datasets missing in target: {len(missing_target_datasets)}")
    for name in missing_target_datasets:
        print(f"  - {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
