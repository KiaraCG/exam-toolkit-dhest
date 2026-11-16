#!/usr/bin/env python3
"""Accuracy per consensus type for majority-vote grades.

    python consensus_breakdown.py grades/harness/majority-vote-baseline
"""
import glob, json, os, sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "grades/harness"
stats = defaultdict(lambda: [0.0, 0.0, 0])          # consensus -> [points, max, n]
for f in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
    if f.endswith("_split.jsonl"):
        continue                                        # strict and split differ only on ties; count each question once
    for line in open(f, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if "consensus" in r:
                s = stats[r["consensus"]]
                s[0] += r["awarded_points"]; s[1] += r["max_points"]; s[2] += 1
print(f"{'consensus':22}{'questions':>10}{'correct':>10}")
for c, (got, mx, n) in sorted(stats.items()):
    print(f"{c:22}{n:>10}{(100 * got / mx if mx else 0):>9.0f}%")
