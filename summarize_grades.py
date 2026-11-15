#!/usr/bin/env python3
"""Table of scores from grade_json.py outputs: one row per exam, one column per model.

    python summarize_grades.py                       # reads grades/*/*/*.jsonl
    python summarize_grades.py grades/signals-and-systems
"""
import glob, json, os, sys

root = sys.argv[1] if len(sys.argv) > 1 else "grades"
scores = {}                                   # (course/exam) -> {model: (got, max, n)}
for f in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)):
    exam, model = os.path.relpath(os.path.dirname(f), root), os.path.splitext(os.path.basename(f))[0]
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    scores.setdefault(exam, {})[model] = (sum(r["awarded_points"] for r in rows),
                                          sum(r["max_points"] for r in rows), len(rows))
models = sorted({m for d in scores.values() for m in d})
print(f"{'exam':28}" + "".join(f"{m:>18}" for m in models))
for exam, d in sorted(scores.items()):
    cells = [f"{d[m][0]:g}/{d[m][1]:g} ({100 * d[m][0] / d[m][1]:.0f}%)" if m in d and d[m][1] else "-" for m in models]
    print(f"{exam:28}" + "".join(f"{c:>18}" for c in cells))
