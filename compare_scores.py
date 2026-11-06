#!/usr/bin/env python3
"""Collect all grading results and write comparison tables (solver x grader) for each exam.

Run from the toolkit root:
    python compare_scores.py                      # all courses
    python compare_scores.py signals-and-systems  # one course

Writes results_tables.html (open it in a browser, select all, copy, paste into Google Docs:
the tables stay tables) and prints a plain-text summary.

Conventions (edit the two dicts below if yours differ):
- Which model TOOK the exam is given by the PDF name: Exam001 = Gemma, Exam002 = Apertus, Exam003 = Nemotron.
- Which model GRADED it is given by the output folder name: evals_apertus -> Apertus, evals_nemotron -> Nemotron,
  any other evals* folder -> Gemma. If an exam has several Gemma folders, the most recent result is used.
"""
import glob, html, json, os, sys
from collections import defaultdict

SOLVERS = {"Exam001": "Gemma", "Exam002": "Apertus", "Exam003": "Nemotron"}
GRADER_KEYWORDS = {"apertus": "Apertus", "nemotron": "Nemotron", "glm": "GLM", "kimi": "Kimi", "qwen": "Qwen", "gptoss": "gpt-oss"}
DEFAULT_GRADER = "Gemma"
GRADER_ORDER = ["Gemma", "Apertus", "Nemotron", "GLM", "Qwen", "gpt-oss"]


def grader_of(folder):
    low = folder.lower()
    for key, name in GRADER_KEYWORDS.items():
        if key in low:
            return name
    return DEFAULT_GRADER


def load():
    course = sys.argv[1] if len(sys.argv) > 1 else "*"
    data = defaultdict(dict)          # (course, exam) -> {(solver, grader): (mtime, result)}
    for f in glob.glob(f"workspaces/{course}/*/evals*/*.json"):
        parts = f.split(os.sep)
        crs, exam, folder, stem = parts[1], parts[2], parts[3], os.path.splitext(parts[4])[0]
        solver = SOLVERS.get(stem, stem)
        key = (solver, grader_of(folder))
        try:
            result = json.load(open(f, encoding="utf-8"))
            result["grades"]
        except (ValueError, KeyError):
            continue
        mtime = os.path.getmtime(f)
        old = data[(crs, exam)].get(key)
        if old is None or mtime > old[0]:
            data[(crs, exam)][key] = (mtime, result)
    return data


def total(result):
    g = result["grades"]
    return sum(x["awarded_points"] for x in g), sum(x["max_points"] for x in g)


def fmt(x):
    return f"{x:g}"


def main():
    data = load()
    if not data:
        sys.exit("No results found under workspaces/*/*/evals*/ - run from the toolkit root.")
    out = ["<html><head><meta charset='utf-8'><style>body{font-family:Arial;font-size:11pt}"
           "table{border-collapse:collapse;margin:6px 0 18px}td,th{border:1px solid #999;padding:3px 8px;text-align:center}"
           "th{background:#eee}td.l{text-align:left}</style></head><body>"]

    # ---- overview: one row per exam and solver, one column per grader ----
    graders = [g for g in GRADER_ORDER if any(k[1] == g for d in data.values() for k in d)]
    out.append("<h2>Overview: score (%) per exam, model taking the exam, and grader</h2><table><tr><th>Exam</th>"
               "<th>Taken by</th>" + "".join(f"<th>Graded by {g}</th>" for g in graders) + "</tr>")
    print("OVERVIEW")
    for (crs, exam) in sorted(data):
        for solver in sorted({k[0] for k in data[(crs, exam)]}, key=lambda s: list(SOLVERS.values()).index(s) if s in SOLVERS.values() else 99):
            cells, line = [], f"{exam:6} taken by {solver:9}"
            for g in graders:
                r = data[(crs, exam)].get((solver, g))
                if r:
                    a, m = total(r[1])
                    cells.append(f"<td>{fmt(a)} / {fmt(m)} ({100 * a / m:.0f}%)</td>")
                    line += f" | {g}: {fmt(a)}/{fmt(m)} ({100 * a / m:.0f}%)"
                else:
                    cells.append("<td>-</td>")
            out.append(f"<tr><td>{exam.upper()}</td><td>{solver}</td>{''.join(cells)}</tr>")
            print(line)
    out.append("</table>")

    # ---- one table per exam and solver: per-item points by each grader ----
    for (crs, exam) in sorted(data):
        for solver in sorted({k[0] for k in data[(crs, exam)]}):
            cols = [g for g in graders if (solver, g) in data[(crs, exam)]]
            res = {g: data[(crs, exam)][(solver, g)][1] for g in cols}
            items = [x["rubric_item"] for x in res[cols[0]]["grades"]]
            maxes = {x["rubric_item"]: x["max_points"] for x in res[cols[0]]["grades"]}
            pts = {g: {x["rubric_item"]: x["awarded_points"] for x in res[g]["grades"]} for g in cols}
            out.append(f"<h3>{exam.upper()} ({crs}), taken by {solver}</h3><table><tr><th>Item</th><th>Max</th>"
                       + "".join(f"<th>{g}</th>" for g in cols) + "<th>Graders agree</th></tr>")
            agree_n = 0
            for it in items:
                vals = [pts[g].get(it) for g in cols]
                same = len(set(vals)) == 1
                agree_n += same
                out.append(f"<tr><td class='l'>{html.escape(it)}</td><td>{fmt(maxes[it])}</td>"
                           + "".join(f"<td>{'-' if v is None else fmt(v)}</td>" for v in vals)
                           + f"<td>{'yes' if same else '<b>no</b>'}</td></tr>")
            tot = [total(res[g]) for g in cols]
            out.append(f"<tr><th>Total</th><th>{fmt(tot[0][1])}</th>"
                       + "".join(f"<th>{fmt(a)} ({100 * a / m:.0f}%)</th>" for a, m in tot)
                       + f"<th>{agree_n}/{len(items)} items</th></tr>")
            flags = [g for g in cols if res[g].get("human_review_recommended")]
            out.append("</table>" + (f"<p>Human review flagged by: {', '.join(flags)}</p>" if flags else ""))

    open("results_tables.html", "w", encoding="utf-8").write("\n".join(out) + "</body></html>")
    print("\nWrote results_tables.html - open it in a browser, select all, copy, paste into Google Docs.")


if __name__ == "__main__":
    main()