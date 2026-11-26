#!/usr/bin/env python3
"""Meta-Harness for MC exams: an open model (proposer) reads earlier harnesses (code, scores, failed questions with
traces) and writes a new harness.py; each candidate is scored by exact match on the SEARCH exams. The proposer never
sees the answer key, only right/wrong per question. Report the best candidate on the TEST exams once, at the end.

    python meta_harness/run_meta.py --search D/2017_autumn.json D/2020_autumn.json --iterations 10
    python meta_harness/run_meta.py --test D/2021_autumn.json            # best candidate on held-out exam
"""
import argparse, glob, json, os, re, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import evaluate, load_harness
from llm import LLM, MODELS

PROPOSER = os.getenv("META_PROPOSER", "RCP-AIaaS/moonshotai/Kimi-K3.1-Pro")

INSTRUCTIONS = """You are optimizing a *harness*: Python code that uses LLMs to answer multiple-choice questions from
university Signals & Systems exams. Your goal: maximize accuracy on the search set while keeping the number of LLM
calls per question reasonable (at most {max_calls}).

Harness interface (write a complete file harness.py):
- define `solve(question, llm) -> str` returning the option letter (e.g. "C").
- `question` is a dict: id, question (text, may contain LaTeX), options (dict letter -> text), image_path (figures are NOT available to you).
- `llm(model, prompt, max_tokens=4000, temperature=None, system=None) -> str` calls a model. Allowed models: {models}.
- `llm.extract_choice(text, options)` extracts the final letter from a reply ('' if none).
- Only the Python standard library may be imported. No network or file access except through llm().
- Calls run in parallel threads: keep solve() free of global mutable state.

Ideas you can combine: better prompts, reasoning, self-verification, several models or samples with voting,
debate/critique rounds, choosing models per question, fallbacks when no letter is found.

Below are all earlier candidates (score = % correct on the search set; calls = LLM calls per question), the code of
the best ones, and questions the best candidate got WRONG with excerpts of its model outputs. The correct answers
are not shown. Study the failures, form a hypothesis, and write ONE new harness that tests it.

Reply with a short rationale (2-4 sentences), then the full file in a single ```python block."""


def candidates(archive):
    out = []
    for d in sorted(glob.glob(os.path.join(archive, "*"))):
        f = os.path.join(d, "score.json")
        if os.path.isfile(f):
            out.append((d, json.load(open(f))))
    return out


def build_prompt(archive, max_calls, n_code=3, n_fail=8):
    cands = candidates(archive)
    ranked = sorted(cands, key=lambda c: (-c[1]["accuracy"], c[1]["calls_per_question"]))
    lines = [INSTRUCTIONS.format(max_calls=max_calls, models=", ".join(MODELS)), "\n## All candidates"]
    for d, s in cands:
        why = open(os.path.join(d, "rationale.md")).read().strip()[:300] if os.path.exists(os.path.join(d, "rationale.md")) else ""
        lines.append(f"- {os.path.basename(d)}: score {s['accuracy']}% {s['per_exam']}, calls {s['calls_per_question']}, "
                     f"errors {s['errors']}. {why}")
    for d, s in ranked[:n_code]:
        lines.append(f"\n## Code of {os.path.basename(d)} (score {s['accuracy']}%)\n```python\n"
                     + open(os.path.join(d, "harness.py")).read() + "\n```")
    best = ranked[0][0]
    wrong = [json.loads(l) for l in open(os.path.join(best, "results.jsonl")) if not json.loads(l)["correct"]][:n_fail]
    exams = {}
    lines.append(f"\n## Questions {os.path.basename(best)} got wrong (answer given, model output excerpt)")
    for r in wrong:
        lines.append(f"\n### {r['exam']} {r['id']}: answered '{r['answer']}'" + (f" (error: {r['error']})" if r["error"] else ""))
        for t in r["trace"][-2:]:
            lines.append(f"[{t['model']}] ...{t['reply'][-800:]}")
    return "\n".join(lines)


def propose(archive, max_calls, out_dir, attempts=3):
    prompt = build_prompt(archive, max_calls)
    open(os.path.join(out_dir, "proposer_prompt.txt"), "w").write(prompt)
    last = "no attempt"
    for i in range(1, attempts + 1):
        if os.getenv("META_FAKE"):
            reply, finish = "Fake proposer: copy the baseline.\n```python\n" + open(os.path.join(HERE, "seeds", "baseline.py")).read() + "\n```", "stop"
        else:
            from llm import _get_client
            try:
                r = _get_client().chat.completions.create(model=PROPOSER, max_tokens=int(os.getenv("META_PROPOSER_TOKENS", 32000)),
                                                           messages=[{"role": "user", "content": prompt}])
                reply, finish = r.choices[0].message.content or "", r.choices[0].finish_reason
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"  proposer attempt {i}/{attempts} failed: {last}", flush=True)
                time.sleep(60 if "429" in str(e) else 20)
                continue
        open(os.path.join(out_dir, f"proposer_reply_{i}.txt"), "w").write(f"[finish_reason: {finish}]\n{reply}")
        blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", reply, flags=re.S)
        blocks = [b for b in blocks if "def solve" in b]
        if blocks:
            rationale = reply[:reply.find("```")].strip()
            return blocks[-1], rationale
        last = f"no code block with solve() (finish_reason={finish}, {len(reply)} chars)"
        print(f"  proposer attempt {i}/{attempts}: {last}", flush=True)
    raise ValueError(f"proposer failed: {last}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default="meta_runs/run1", help="folder with all candidates (git-ignore it)")
    ap.add_argument("--search", nargs="+", help="search exams (exam JSON files)")
    ap.add_argument("--test", nargs="+", help="held-out exams: evaluate the best candidate once and stop")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--limit", type=int, help="first N questions per search exam (cheaper search)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.archive, exist_ok=True)
    max_calls = LLM.max_calls_per_question

    if a.test:
        d, s = sorted(candidates(a.archive), key=lambda c: (-c[1]["accuracy"], c[1]["calls_per_question"]))[0]
        out = os.path.join(a.archive + "_test", os.path.basename(d))
        os.makedirs(out, exist_ok=True)
        shutil.copy(os.path.join(d, "harness.py"), out)
        print(f"Best on search: {os.path.basename(d)} ({s['accuracy']}%). Test:", json.dumps(evaluate(out, a.test, a.workers)))
        return

    for seed in sorted(glob.glob(os.path.join(HERE, "seeds", "*.py"))):            # seeds = hand-made harnesses
        d = os.path.join(a.archive, "000_" + os.path.basename(seed)[:-3])
        if not os.path.exists(os.path.join(d, "score.json")):
            os.makedirs(d, exist_ok=True)
            shutil.copy(seed, os.path.join(d, "harness.py"))
            print("seed", os.path.basename(d), json.dumps(evaluate(d, a.search, a.workers, a.limit)), flush=True)

    start = len(glob.glob(os.path.join(a.archive, "[0-9][0-9][0-9]_iter*")))
    for i in range(start + 1, start + a.iterations + 1):
        d = os.path.join(a.archive, f"{i:03d}_iter")
        os.makedirs(d, exist_ok=True)
        try:
            code, rationale = propose(a.archive, max_calls, d)
            open(os.path.join(d, "harness.py"), "w").write(code)
            open(os.path.join(d, "rationale.md"), "w").write(rationale)
            load_harness(os.path.join(d, "harness.py"))                              # fails fast on broken code
            s = evaluate(d, a.search, a.workers, a.limit)
            print(f"iter {i}: {s['accuracy']}% {s['per_exam']} calls {s['calls_per_question']} | {rationale[:120]}", flush=True)
        except Exception as e:
            open(os.path.join(d, "failed.txt"), "w").write(f"{type(e).__name__}: {e}")
            print(f"iter {i}: FAILED ({type(e).__name__}: {str(e)[:150]})", flush=True)

    print("\nLeaderboard (search set):")
    for d, s in sorted(candidates(a.archive), key=lambda c: -c[1]["accuracy"]):
        print(f"  {s['accuracy']:5.1f}%  calls {s['calls_per_question']:4}  {os.path.basename(d)}")


if __name__ == "__main__":
    main()