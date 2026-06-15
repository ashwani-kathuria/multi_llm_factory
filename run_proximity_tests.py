"""
run_proximity_tests.py
======================
Command-line test runner for the Topic-Aware Self-Correction Proximity metric.

Reads test cases from CSV/TXT batch files and runs proximity analysis on each
ReasoningText, reporting TotalHedges, ResolvedHedges, and TRUR.

Usage examples
--------------
# Single test case (searches all batch files)
python run_proximity_tests.py TC001

# Multiple specific test cases
python run_proximity_tests.py TC001 TC025 TC050

# All 100 test cases
python run_proximity_tests.py --all

# Entire batch
python run_proximity_tests.py --batch 1
python run_proximity_tests.py --batch 1 2 3

# Verbose: show per-hedge breakdown for each case
python run_proximity_tests.py TC001 --verbose

# Save report to file
python run_proximity_tests.py --all --output report.txt
"""

import argparse
import csv
import io
import os
import sys
import textwrap
import time
from typing import Dict, List, Optional

# ── Make sure the project root is importable ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from proximity_metric import calculate_proximity_metrics   # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
BATCH_FILES: Dict[int, str] = {
    1: "testCases_Batch_1.txt",
    2: "testCases_Batch_2.txt",
    3: "testCases_Batch_3.txt",
    4: "testCases_Batch_4.txt",
}
EXPECTED_COLUMNS = [
    "TestCaseID", "Category", "CertaintyLevel",
    "ConfidenceTrajectory", "ExpectedLabel", "ReasoningText", "Explanation",
]


# ── CSV loading ───────────────────────────────────────────────────────────────
def _load_batch(batch_number: int) -> List[dict]:
    """Load and parse a single batch file, returning a list of row dicts."""
    filename = BATCH_FILES.get(batch_number)
    if not filename:
        raise ValueError(f"Unknown batch number: {batch_number}. Valid: 1–4.")

    filepath = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Batch file not found: '{filepath}'")

    rows: List[dict] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_test_cases(
    batch_numbers: Optional[List[int]] = None,
) -> List[dict]:
    """
    Load test cases from one or more batch files.
    If batch_numbers is None, all four batches are loaded.
    """
    batches = batch_numbers if batch_numbers else list(BATCH_FILES.keys())
    all_rows: List[dict] = []
    for b in batches:
        all_rows.extend(_load_batch(b))
    return all_rows


def filter_test_cases(
    all_cases: List[dict],
    ids: Optional[List[str]] = None,
) -> List[dict]:
    """Return only the rows whose TestCaseID is in *ids* (None → return all)."""
    if ids is None:
        return all_cases
    id_set = {tc_id.upper() for tc_id in ids}
    found = [r for r in all_cases if r["TestCaseID"].upper() in id_set]
    # Warn about IDs that were requested but not found
    found_ids = {r["TestCaseID"].upper() for r in found}
    missing = id_set - found_ids
    if missing:
        print(f"[WARNING] Test case ID(s) not found: {', '.join(sorted(missing))}\n",
              file=sys.stderr)
    return found


# ── Runner ────────────────────────────────────────────────────────────────────
def run_test_case(row: dict, verbose: bool = False) -> dict:
    """
    Execute the proximity metric on one test case row.

    Returns a result dict with the original fields plus:
      total_hedges, resolved_hedges, unresolved_hedges, trur, weighted_trur,
      matches, elapsed_ms
    """
    reasoning_text = row.get("ReasoningText", "").strip()

    t0 = time.perf_counter()
    metrics = calculate_proximity_metrics(reasoning_text)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "TestCaseID":           row["TestCaseID"],
        "Category":             row.get("Category", ""),
        "CertaintyLevel":       row.get("CertaintyLevel", ""),
        "ConfidenceTrajectory": row.get("ConfidenceTrajectory", ""),
        "ExpectedLabel":        row.get("ExpectedLabel", ""),
        # proximity output
        "total_hedges":          metrics["total_hedges"],
        "resolved_hedges":       metrics["resolved_hedges"],
        "unresolved_hedges":     metrics["unresolved_hedges"],
        "trur":                  metrics["trur"],
        "weighted_trur":         metrics["weighted_trur"],
        "late_unresolved_ratio": metrics.get("late_unresolved_ratio", 0.0),
        "conclusion_finality":   metrics.get("conclusion_finality", 0.0),
        "predicted_certainty":   metrics.get("predicted_certainty", "?"),
        "matches":               metrics["matches"],
        "elapsed_ms":            round(elapsed_ms, 1),
    }


# ── Display helpers ───────────────────────────────────────────────────────────
_COL_WIDTHS = {
    "TestCaseID":     10,
    "Category":       12,
    "CertaintyLevel": 14,
    "ExpectedLabel":  14,
    "TotalHedges":     12,
    "ResolvedHedges":  15,
    "TRUR":           10,
    "WeightedTRUR":   14,
    "Time(ms)":        9,
}
_HEADER_ROW = (
    f"{'TestCaseID':<10}  "
    f"{'Category':<12}  "
    f"{'Certainty':<14}  "
    f"{'Expected':<14}  "
    f"{'TotalHedges':>11}  "
    f"{'Resolved':>8}  "
    f"{'TRUR':>8}  "
    f"{'WtdTRUR':>8}  "
    f"{'Predicted':<12}  "
    f"{'ms':>7}"
)
_SEP = "-" * len(_HEADER_ROW)


def _format_result_row(r: dict) -> str:
    predicted = r.get("predicted_certainty", "?")
    return (
        f"{r['TestCaseID']:<10}  "
        f"{r['Category']:<12}  "
        f"{r['CertaintyLevel']:<14}  "
        f"{r['ExpectedLabel']:<14}  "
        f"{r['total_hedges']:>11}  "
        f"{r['resolved_hedges']:>8}  "
        f"{r['trur']:>7.1%}  "
        f"{r['weighted_trur']:>7.1%}  "
        f"{predicted:<12}  "
        f"{r['elapsed_ms']:>7.1f}"
    )


def _print_verbose_detail(r: dict) -> None:
    """Print per-hedge breakdown for a single result."""
    print(f"\n  Detail for {r['TestCaseID']}:")
    for m in r["matches"]:
        status = "✓ RESOLVED" if m["resolved"] else "✗ unresolved"
        print(f"    [{m['id']}] {status}")
        print(f"      Sentence     : {textwrap.shorten(m['sentence'], 90)}")
        print(f"      Keyword      : \"{m['matched_keyword']}\"")
        print(f"      Subject      : {m['subject']}")
        if m.get("matched_verification"):
            print(f"      Matched verif: {m['matched_verification']}  "
                  f"(sim={m['subject_similarity']:.4f}, "
                  f"prox={m['proximity_score']:.4f}, "
                  f"score={m['match_score']:.4f})")
        print(f"      Eff. weight  : {m['effective_weight']:.4f}")
    print()


def _print_group_breakdown(
    results: List[dict],
    group_key: str,
    label: str,
    key_order: Optional[List[str]] = None,
    out=sys.stdout,
) -> None:
    """
    Print a compact breakdown table grouping *results* by *group_key*.

    key_order, when supplied, determines the row order (useful for e.g.
    High → Medium → Low). Groups not in key_order are appended alphabetically.
    """
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for r in results:
        groups[r.get(group_key, "Unknown")].append(r)

    if not groups:
        return

    # Determine row order
    ordered_keys: List[str] = []
    if key_order:
        ordered_keys = [k for k in key_order if k in groups]
        ordered_keys += sorted(k for k in groups if k not in key_order)
    else:
        ordered_keys = sorted(groups.keys())

    col_g  = max(len(label), max(len(k) for k in ordered_keys)) + 2
    header = (
        f"\n  {label}\n"
        f"  {'Group':<{col_g}}  {'Cases':>5}  {'AvgHedges':>9}  "
        f"{'AvgResolved':>11}  {'AvgTRUR':>7}  {'AvgWtdTRUR':>10}"
    )
    sep = "  " + "-" * (col_g + 52)

    print(header, file=out)
    print(sep, file=out)

    for key in ordered_keys:
        grp = groups[key]
        n   = len(grp)
        avg_h  = sum(r["total_hedges"]    for r in grp) / n
        avg_rs = sum(r["resolved_hedges"] for r in grp) / n
        avg_t  = sum(r["trur"]            for r in grp) / n
        avg_wt = sum(r["weighted_trur"]   for r in grp) / n
        print(
            f"  {key:<{col_g}}  {n:>5}  {avg_h:>9.2f}  "
            f"{avg_rs:>11.2f}  {avg_t:>7.1%}  {avg_wt:>10.1%}",
            file=out,
        )

    print(sep, file=out)


def print_results(results: List[dict], verbose: bool = False,
                  out=sys.stdout) -> None:
    """Render the full results table."""
    print(_HEADER_ROW, file=out)
    print(_SEP, file=out)

    for r in results:
        print(_format_result_row(r), file=out)
        if verbose:
            _print_verbose_detail(r)

    print(_SEP, file=out)

    # ── Summary statistics ────────────────────────────────────────────────────
    total_cases   = len(results)
    avg_hedges    = sum(r["total_hedges"]    for r in results) / total_cases if total_cases else 0
    avg_resolved  = sum(r["resolved_hedges"] for r in results) / total_cases if total_cases else 0
    avg_trur      = sum(r["trur"]            for r in results) / total_cases if total_cases else 0
    avg_wtrur     = sum(r["weighted_trur"]   for r in results) / total_cases if total_cases else 0
    total_time_ms = sum(r["elapsed_ms"]      for r in results)

    zero_hedge    = sum(1 for r in results if r["total_hedges"] == 0)
    full_resolved = sum(1 for r in results if r["total_hedges"] > 0
                        and r["resolved_hedges"] == r["total_hedges"])

    print(f"\nSummary ({total_cases} test case(s))", file=out)
    print(f"  Avg hedges per case : {avg_hedges:.2f}", file=out)
    print(f"  Avg resolved        : {avg_resolved:.2f}", file=out)
    print(f"  Avg TRUR            : {avg_trur:.1%}", file=out)
    print(f"  Avg Weighted TRUR   : {avg_wtrur:.1%}", file=out)
    print(f"  Cases with 0 hedges : {zero_hedge}", file=out)
    print(f"  Fully resolved cases: {full_resolved}", file=out)
    print(f"  Total wall time     : {total_time_ms/1000:.2f}s", file=out)

    # ── Breakdown by CertaintyLevel ───────────────────────────────────────────
    _print_group_breakdown(results, group_key="CertaintyLevel",
                           label="By Certainty Level",
                           key_order=["High", "Medium", "Low"],
                           out=out)

    # ── Breakdown by Category ─────────────────────────────────────────────────
    _print_group_breakdown(results, group_key="Category",
                           label="By Category",
                           out=out)


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_proximity_tests",
        description=(
            "Run the Topic-Aware Self-Correction Proximity metric "
            "against one or more test cases from the batch files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run_proximity_tests.py TC001
              python run_proximity_tests.py TC001 TC025 TC050
              python run_proximity_tests.py --all
              python run_proximity_tests.py --batch 1
              python run_proximity_tests.py --batch 2 3 --verbose
              python run_proximity_tests.py --all --output report.txt
        """),
    )

    # Positional: zero or more specific test case IDs
    parser.add_argument(
        "test_case_ids",
        nargs="*",
        metavar="TCXXX",
        help="One or more test case IDs to run (e.g. TC001 TC025).",
    )
    # Mutually exclusive selection flags
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument(
        "--all", "-all",
        action="store_true",
        help="Run all 100 test cases across all four batch files.",
    )
    sel.add_argument(
        "--batch", "-batch",
        nargs="+",
        type=int,
        metavar="N",
        choices=[1, 2, 3, 4],
        help="Run all test cases from the specified batch(es) (1–4).",
    )
    # Options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-hedge breakdown for each test case.",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Write results to FILE in addition to stdout.",
    )
    return parser


def main() -> None:
    # Fix Windows console encoding for Unicode symbols (✓ / ✗)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args()

    # ── Validate that at least one selection was made ─────────────────────────
    if not args.all and not args.batch and not args.test_case_ids:
        parser.error(
            "Specify one or more test case IDs, --all, or --batch N.\n"
            "Run 'python run_proximity_tests.py --help' for usage."
        )

    # ── Load the relevant batch files ─────────────────────────────────────────
    if args.all:
        print("Loading all 4 batch files (100 test cases)…")
        all_cases = load_test_cases()
        selected = all_cases
    elif args.batch:
        print(f"Loading batch(es): {args.batch}…")
        all_cases = load_test_cases(batch_numbers=args.batch)
        selected = all_cases
    else:
        # Specific IDs — need to search across all batches
        all_cases = load_test_cases()
        selected = filter_test_cases(all_cases, ids=args.test_case_ids)

    if not selected:
        print("No matching test cases found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(selected)} test case(s)…\n")

    # ── Run each test case ────────────────────────────────────────────────────
    results: List[dict] = []
    for i, row in enumerate(selected, 1):
        tc_id = row.get("TestCaseID", f"row-{i}")
        print(f"  [{i:>3}/{len(selected)}] {tc_id} … ", end="", flush=True)
        try:
            result = run_test_case(row, verbose=args.verbose)
            results.append(result)
            print(
                f"hedges={result['total_hedges']}  "
                f"resolved={result['resolved_hedges']}  "
                f"TRUR={result['trur']:.1%}  "
                f"({result['elapsed_ms']:.0f}ms)"
            )
        except Exception as exc:
            print(f"ERROR — {exc}", file=sys.stderr)
            # Append a placeholder so row count stays consistent
            results.append({
                "TestCaseID": tc_id,
                "Category": row.get("Category", ""),
                "CertaintyLevel": row.get("CertaintyLevel", ""),
                "ConfidenceTrajectory": row.get("ConfidenceTrajectory", ""),
                "ExpectedLabel": row.get("ExpectedLabel", ""),
                "total_hedges": -1,
                "resolved_hedges": -1,
                "unresolved_hedges": -1,
                "trur": 0.0,
                "weighted_trur": 0.0,
                "matches": [],
                "elapsed_ms": 0.0,
            })

    # ── Print results table ───────────────────────────────────────────────────
    print()
    print_results(results, verbose=args.verbose)

    # ── Optional file output ──────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            print_results(results, verbose=args.verbose, out=fh)
        print(f"\n📝 Results also saved to: '{args.output}'")


if __name__ == "__main__":
    main()
