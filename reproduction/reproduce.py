#!/usr/bin/env python3
"""Recompute the included analyses and compare them with the expected results.

The package is self-contained and needs no network access.  Code and inputs are
copied into runs/<UTC timestamp>/ so the shipped tree is never modified.
"""
from pathlib import Path
import argparse
import datetime
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STAGES = ['code', 'data']
TOLERANCE = 0.0
CSV_DECIMALS = 9

IGNORE = set(['claim_tag', 'code_sha256', 'elapsed_query_only_s', 'elapsed_s', 'elapsed_seconds', 'enumeration_query_only_s', 'execution', 'input_readback_sha256', 'input_sha256', 'log_sha256', 'median_exact_subset_score_seconds', 'median_seconds', 'numpy', 'peak_rss_bytes_macos', 'prefix_query_only_s', 'python', 'query_only_s', 'readback_sha256', 'root_code_sha256', 'root_results_sha256', 'score_all_subsets_seconds', 'script_sha256', 'seconds', 'software', 'source_data_hashes_from_intake_receipt', 'source_files', 'source_receipt_sha256', 'source_sha256', 'sources', 'sqlite', 'time_utc', 'timing', 'timings_seconds', 'total_elapsed_s', 'tzpath'])
TIMING = re.compile(r"(^|_)(seconds?|secs?|elapsed|timing|timings|duration|"
                    r"runtime|wall|latency|throughput_per_sec)($|_)", re.I)
DROPPED = set()


def semantic(x):
    """Strip execution records, environment fields and wall-clock timings.

    What remains is the scientific content: integer counts, memberships, and
    values at printed precision.  Names of the fields actually dropped are
    collected in DROPPED so CHECK.json can report them.
    """
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in IGNORE or TIMING.search(str(k)):
                DROPPED.add(str(k))
            else:
                out[k] = semantic(v)
        return out
    if isinstance(x, list):
        return [semantic(v) for v in x]
    if isinstance(x, float) and x != x:
        # A SUM over an empty selection arrives as NaN.  Canonicalise it to text
        # so the canonical form stays JSON-serialisable and compares equal to
        # itself (NaN != NaN in IEEE arithmetic, which would always mismatch).
        return "NaN"
    if isinstance(x, str) and x.startswith("/"):
        return x.rsplit("/", 1)[-1]
    return x

def cell(v, decimals=9):
    """Canonical form of one CSV cell: numbers rounded to `decimals` decimal
    places, everything else left as text.  A non-finite number is left as text,
    since NaN never equals itself and neither is JSON-serialisable.

    The default (9) sits below every pipeline's own write precision.  A package
    whose reference carries low-order solver noise that is large relative to a
    small printed value can compare at the coarser precision its paper prints by
    passing a smaller `decimals`; rounding to fewer places is strictly more
    lenient, so such a package cannot be made to pass a value its finer-precision
    sibling would reject."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ["s", v.strip()]
    if f != f or f in (float("inf"), float("-inf")):
        return ["s", v.strip()]
    return ["#", round(f, decimals)]


def table_row(r, decimals=9):
    return [cell(c, decimals) for c in r]


def load_result(path, decimals=9):
    """Canonical form of a result file.

    .json  -> the semantic content
    .csv   -> {"__table__": header + value-sorted row multiset}, since row
              order is not part of the comparison contract; numeric cells are
              rounded to `decimals` decimal places first
    other  -> raw text
    """
    text = path.read_text()
    if path.suffix == ".csv":
        import csv as _csv
        import io
        rows = [r for r in _csv.reader(io.StringIO(text)) if r]
        if not rows:
            return {"__table__": {"head": [], "rows": []}}
        head = [h.strip() for h in rows[0]]
        body = sorted(table_row(r, decimals) for r in rows[1:])
        return {"__table__": {"head": head, "rows": body}}
    if path.suffix == ".json":
        return semantic(json.loads(text))
    return text


def same(a, b, rtol):
    """Deep comparison.  Integers and text must agree exactly; floats are
    allowed to differ in their low-order bits, which the papers' own READMEs
    permit and which BLAS thread-reduction order can change."""
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], rtol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or abs(a - b) <= rtol * max(abs(a), abs(b))
    return a == b


def first_difference(a, b, rtol, path=""):
    """Locate the first mismatch, for a useful failure message."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return path + "/" + str(k) + ": missing from actual"
            if k not in b:
                return path + "/" + str(k) + ": unexpected in actual"
            d = first_difference(a[k], b[k], rtol, path + "/" + str(k))
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return path + ": length " + str(len(a)) + " != " + str(len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, rtol, path + "[" + str(i) + "]")
            if d:
                return d
        return None
    if not same(a, b, rtol):
        return path + ": " + repr(a)[:120] + " != " + repr(b)[:120]
    return None

def steps(py):
    yield "queries", [py, "-B", "code/run_queries.py",
                      "--data", "data/official_sample", "--out", "work/queries"]
    yield "sqlite_baseline", [py, "-B", "code/sqlite_semantic_baseline.py",
                              "--data", "data/official_sample",
                              "--results", "work/queries", "--code", "code",
                              "--out", "results/primary"]
    yield "targeted_membership", [py, "-B", "code/sqlite_targeted_membership.py",
                                  "--data", "data/official_sample",
                                  "--out", "results/primary"]
    yield "joint_verification", [py, "-B", "code/joint_verification_exhaustive.py",
                                 "--source",
                                 "results/primary/targeted_variable_membership.json",
                                 "--out", "results/joint_verification"]
    yield "transfer", [py, "-B", "code/transfer.py",
                       "--out", "results/transfer"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-limit", type=int, default=120,
                        help="CPU seconds per child process on POSIX systems")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE,
                        help="relative tolerance for floating point values; "
                             "0 requires exact equality")
    a = parser.parse_args()

    run = ROOT / "runs" / datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run.mkdir(parents=True)
    for folder in STAGES:
        shutil.copytree(ROOT / folder, run / folder)

    processes = []

    def call(label, args):
        # The papers' own drivers pre-create their output directories, and the
        # run copy already holds them.  Re-assert every staged directory before
        # each step so a step can never die on a directory that went missing
        # between steps; this is a no-op whenever the copy is intact.
        for folder in STAGES:
            (run / folder).mkdir(parents=True, exist_ok=True)

        def light():
            os.nice(15)
            resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_limit, a.cpu_limit))
        start = time.monotonic()
        r = subprocess.run(list(map(str, args)), cwd=run, capture_output=True,
                           timeout=max(60, a.cpu_limit * 4), preexec_fn=light,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        (run / (label + ".stdout")).write_bytes(r.stdout)
        (run / (label + ".stderr")).write_bytes(r.stderr)
        processes.append({"label": label, "exit": r.returncode,
                          "seconds": time.monotonic() - start})
        (run / "processes.json").write_text(json.dumps(processes, indent=2) + "\n")
        if r.returncode:
            raise RuntimeError(label + " failed with exit " + str(r.returncode)
                               + "\n" + r.stderr.decode("utf-8", "replace")[-4000:])
        print(label + ": PASS", flush=True)

    for label, argv in steps(sys.executable):
        call(label, argv)

    expected = json.loads((ROOT / "expected.json").read_text())
    failures = []
    for name in sorted(expected):
        actual = load_result(run / name, CSV_DECIMALS)
        if not same(actual, expected[name], a.tolerance):
            failures.append(name + "  "
                            + str(first_difference(actual, expected[name], a.tolerance)))
    if failures:
        raise AssertionError("Expected result mismatch in " + str(len(failures))
                             + " file(s):\n  " + "\n  ".join(failures[:20]))

    check = {"status": "PASS", "expected_result_files": len(expected),
             "tolerance": a.tolerance, "ignored_fields": sorted(DROPPED),
             "processes": processes, "run": str(run)}
    (run / "CHECK.json").write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check))


if __name__ == "__main__":
    main()
