#!/usr/bin/env python3
"""Backward-compatible wrapper for the unified figures script."""

from __future__ import annotations

import argparse
from pathlib import Path

from figures import run_pangram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=None)
    parser.add_argument("--collections", nargs="*", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--output", type=Path, default=Path("results/pangram_distributions.png"))
    args = parser.parse_args()
    run_pangram(args)


if __name__ == "__main__":
    main()

