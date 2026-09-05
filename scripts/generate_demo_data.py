"""Regenerate the synthetic demo datasets and their ground truth file.

Usage::

    python scripts/generate_demo_data.py [--output data/demo]

The generator is seeded, so running it twice produces byte-identical files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dqcopilot.demodata import write_demo_data

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "demo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination directory (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    ground_truth = write_demo_data(args.output)
    files = sorted(path.name for path in args.output.iterdir())

    print(f"Wrote {len(files)} file(s) to {args.output}:")
    for name in files:
        print(f"  - {name}")
    print(f"\nGround truth: {ground_truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
