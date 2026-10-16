#!/usr/bin/env python3
"""Cross-domain budget comparison and global-edition controls."""
from pathlib import Path
import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from code import experiment as E
from code.controls import run

a.out.mkdir(parents=True, exist_ok=True)
E.m2(a.out)
run(a.out)
