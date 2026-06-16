"""Count binary prediction accuracy from a proximity test report file."""

import sys

REPORT = sys.argv[1] if len(sys.argv) > 1 else "full_report_v4.txt"

EXPECTED_LABELS = [
    "Unresolved Uncertainty",
    "Resolved Uncertainty",
    "Partially Resolved",
    "Certain",
]

PREDICTED_LABELS = ["Certain", "Uncertain"]


def expected_binary(label: str) -> str | None:
    """Map ExpectedLabel to binary ground truth.

    Partially Resolved is genuinely in the middle — we accept either prediction
    (return None = don't count for accuracy).
    """
    if label in ("Certain", "Resolved Uncertainty"):
        return "Certain"
    if label == "Unresolved Uncertainty":
        return "Uncertain"
    return None   # Partially Resolved — either is acceptable


def extract_fields(line: str):
    line = line.rstrip()
    if not line.startswith("TC"):
        return None
    parts = line.split()
    if len(parts) < 3:
        return None
    tc_id    = parts[0]
    predicted = parts[-2]
    if predicted not in PREDICTED_LABELS:
        return None
    expected = None
    for label in EXPECTED_LABELS:
        if label in line:
            expected = label
            break
    return tc_id, expected, predicted


def main():
    with open(REPORT, encoding="utf-8") as fh:
        lines = fh.readlines()

    rows = []
    for line in lines:
        result = extract_fields(line)
        if result:
            rows.append(result)

    cats = {lbl: {"correct": 0, "total": 0, "breakdown": {}} for lbl in EXPECTED_LABELS}

    total_correct = 0
    total_counted = 0   # excludes Partially Resolved (neither right nor wrong)
    misses = []

    for tc_id, expected, predicted in rows:
        truth = expected_binary(expected)

        if expected in cats:
            cats[expected]["total"] += 1
            cats[expected]["breakdown"][predicted] = \
                cats[expected]["breakdown"].get(predicted, 0) + 1

        if truth is None:
            # Partially Resolved — both labels acceptable; count as correct
            cats[expected]["correct"] += 1
            total_correct += 1
            total_counted += 1
            continue

        total_counted += 1
        if predicted == truth:
            total_correct += 1
            cats[expected]["correct"] += 1
        else:
            misses.append((tc_id, expected, f"expected={truth}", f"got={predicted}"))

    n = len(rows)
    print(f"\nPrediction Accuracy (Binary) — {REPORT}")
    print("=" * 65)
    print(f"  Total cases   : {n}")
    print(f"  Counted cases : {total_counted}  (Partially Resolved counts as correct for either)")
    print(f"  Correct       : {total_correct}  ({total_correct/total_counted*100:.1f}%)")
    print(f"  Wrong         : {total_counted - total_correct}  ({(total_counted-total_correct)/total_counted*100:.1f}%)")

    print(f"\n{'Category':<28}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}  Breakdown")
    print("-" * 95)
    for lbl in EXPECTED_LABELS:
        d = cats[lbl]
        acc = d["correct"] / d["total"] * 100 if d["total"] else 0
        bd  = "  |  ".join(f"{k}:{v}" for k, v in sorted(d["breakdown"].items()))
        note = " (accept any)" if lbl == "Partially Resolved" else ""
        print(f"{lbl:<28}  {d['correct']:>7}  {d['total']:>5}  {acc:>7.1f}%  {bd}{note}")

    print(f"\nMiscategorised cases ({len(misses)}):")
    print(f"  {'TestCaseID':<12}  {'ExpectedLabel':<28}  {'Truth':<18}  {'Prediction'}")
    print("  " + "-" * 68)
    for tc_id, expected, truth, got in misses:
        print(f"  {tc_id:<12}  {expected:<28}  {truth:<18}  {got}")


if __name__ == "__main__":
    main()
