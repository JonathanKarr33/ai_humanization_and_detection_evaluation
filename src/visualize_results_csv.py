#!/usr/bin/env python3
"""Backward-compatible wrapper for the unified figures script."""

from __future__ import annotations

import argparse
from pathlib import Path

from figures import run_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"))
    parser.add_argument("--max-domains", type=int, default=16)
    args = parser.parse_args()
    run_results(args)


if __name__ == "__main__":
    main()

