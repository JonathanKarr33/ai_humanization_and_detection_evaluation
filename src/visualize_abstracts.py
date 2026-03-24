#!/usr/bin/env python3
"""Backward-compatible wrapper for the unified figures script."""

from __future__ import annotations

import argparse
from pathlib import Path

from figures import run_abstracts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=None)
    parser.add_argument("--min-words", type=int, default=25)
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    run_abstracts(args)


if __name__ == "__main__":
    main()

