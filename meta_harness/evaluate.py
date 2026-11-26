"""Run one harness on MC exams and score it by exact match. The harness never sees the answer key."""
import importlib.util, json, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from llm import LLM, start_trace, get_trace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # toolkit root: grade_json.py
from grade_json import choices                                                     # same matching as the grader


def load_harness(path):
    spec = importlib.util.spec_from_file_location(f"harness_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "solve"):
        raise ValueError("harness.py has no solve(question, llm)")
    return mod


def questions_of(exam_path, limit=None):
    exam = json.load(open(exam_path, encoding="utf-8"))
    out = []
    for q in exam["questions"]:
        if not q.get("options"):
            continue
        public = {k: v for k, v in q.items() if k not in ("answer", "solution", "rubric")}   # hide the key
        out.append((os.path.basename(exam_path).replace(".json", ""), public, q["answer"]))
    return out[:limit] if limit else out


def evaluate(cand_dir, exams, workers=4, limit=None):
    mod = load_harness(os.path.join(cand_dir, "harness.py"))
    llm = LLM()
    items = [x for e in exams for x in questions_of(e, limit)]

    def run(item):
        exam, q, key = item
        start_trace()
        t = time.monotonic()
        try:
            ans, err = str(mod.solve(dict(q), llm) or ""), None
        except Exception as e:
            ans, err = "", f"{type(e).__name__}: {e}"[:300]
        trace, calls = get_trace()
        ok = bool(ans) and choices(ans, q["options"]) == choices(key, q["options"])
        return {"exam": exam, "id": q["id"], "answer": ans, "correct": ok, "error": err,
                "calls": calls, "seconds": round(time.monotonic() - t, 1), "trace": trace}

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(run, items))
    with open(os.path.join(cand_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per_exam = {}
    for r in results:
        per_exam.setdefault(r["exam"], []).append(r["correct"])
    score = {"accuracy": round(100 * sum(r["correct"] for r in results) / len(results), 1),
             "per_exam": {e: round(100 * sum(v) / len(v), 1) for e, v in per_exam.items()},
             "n": len(results), "errors": sum(1 for r in results if r["error"]),
             "calls_per_question": round(sum(r["calls"] for r in results) / len(results), 1)}
    json.dump(score, open(os.path.join(cand_dir, "score.json"), "w"), indent=2)
    return score


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate one candidate folder (with harness.py) on MC exams.")
    ap.add_argument("candidate")
    ap.add_argument("--exams", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="first N questions per exam (quick tests)")
    a = ap.parse_args()
    print(json.dumps(evaluate(a.candidate, a.exams, a.workers, a.limit), indent=2))
