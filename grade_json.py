#!/usr/bin/env python3
"""Grade a model's exam answers (JSON) against the official solution (JSON).

- single-choice / multiple-choice / number: exact match with the key, no LLM call.
- free-text: one LLM call per question (open Swiss AI model only), up to --attempts tries each.
- Every graded question is appended to a .jsonl file at once, so an interrupted run
  resumes where it stopped: questions already in the file are skipped.

Usage (from the toolkit root, venv active, key exported in the shell):
    python grade_json.py \
        --exam      ../aiscientist-dhest/data/signals-and-systems/2017_autumn.json \
        --answers   llm_output/signals-and-systems/gemma/solutions_2017_autumn.json \
        --out       grades/signals-and-systems/hs17/gemma.jsonl

Keep data/, llm_output/ and grades/ git-ignored: they contain exam material and results.
"""
import argparse, json, math, os, re, sys, time

SWISS_AI = "https://api.swissai.svc.cscs.ch/v1"   # rubric/solution may only go to open models
DEFAULT_GRADER = "CSCS-Inference/google/gemma-4-31B-it"
EXACT_TYPES = {"single-choice", "multiple-choice", "number"}

PROMPT = """You are a strict, fair university exam grader. Grade the candidate's answer to ONE exam question.

Question ({points} points):
{question}

Official reference solution (may include the official point allocation for partial credit):
{solution}

Candidate's answer:
{answer}

Rules:
- Award between 0 and {points} points, in steps of {step}. Follow the official point allocation when given.
- Give credit for correct alternative methods; give no credit for unsupported final results unless the question says no justification is needed.
- Judge only what is written. Do not reward length.

Return ONLY one JSON object, no markdown:
{{"awarded_points": <number>, "explanation": "<one or two sentences citing what is right or wrong>"}}"""


# ---------- loading ----------
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path):
    """Accept {"answers": [{"id", "answer"}, ...]}, a plain list of those, or {"q1": answer, ...}."""
    d = load_json(path)
    items = d.get("answers", d) if isinstance(d, dict) else d
    if isinstance(items, dict):
        return {str(k): v for k, v in items.items()}
    out = {}
    for x in items:
        if isinstance(x, dict) and "id" in x:
            out[str(x["id"])] = x.get("answer")
    return out


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done[r["id"]] = r
                    except (ValueError, KeyError):
                        pass          # a half-written last line after a crash is ignored and regraded
    return done


def append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------- exact matching ----------
def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(" .")


def option_map(options):
    """Options as {"A": text, ...} (Kiara's format) or [text, ...] (letters A, B, ... by position)."""
    if isinstance(options, dict):
        return {str(k).strip().upper(): v for k, v in options.items()}
    if isinstance(options, list):
        return {chr(ord("A") + i): v for i, v in enumerate(options)}
    return {}


def choices(ans, options):
    """Turn an answer (letter, 'C)', 'A, C', option text, or a list of these) into a set of option letters.
    Falls back to the normalized text when it cannot be mapped."""
    om = option_map(options)
    by_text = {norm(v): k for k, v in om.items()}
    out = set()
    for a in (ans if isinstance(ans, list) else [ans]):
        s = str(a).strip()
        if norm(s) in by_text:                                  # full option text
            out.add(by_text[norm(s)]); continue
        s = re.sub(r"^(?:final\s+)?answer\s*[:=]?\s*", "", s, flags=re.I)
        parts = [x for x in re.split(r"\s*(?:,|;|/|\band\b|\s)\s*", s) if x]
        letters = [re.fullmatch(r"[\(\[]?([A-Za-z])[\)\].:]?", x) for x in parts]
        if om and parts and all(m and m.group(1).upper() in om for m in letters):
            out |= {m.group(1).upper() for m in letters}
        else:
            out.add(norm(s))
    return out


def to_number(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    m = re.search(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", str(x).replace(",", "."))
    return float(m.group()) if m else None


def exact_grade(q, sol, ans):
    pts = sol.get("points", q.get("points", 1))
    key = sol.get("correctAnswer", sol.get("correctAnswers", sol.get("answer")))
    t = q.get("type")
    opts = q.get("options")
    if t == "number":
        a, k = to_number(ans), to_number(key)
        ok = a is not None and k is not None and math.isclose(a, k, rel_tol=1e-2, abs_tol=1e-6)
    else:                                   # single- or multiple-choice: compare sets of option letters
        ok = choices(ans, opts) == choices(key, opts)
    return {"awarded_points": pts if ok else 0, "correct": ok, "key": key}


# ---------- LLM grading ----------
def extract_json(raw):
    raw = (raw or "").strip()
    a, b = raw.find("{"), raw.rfind("}")
    if a == -1 or b <= a:
        raise ValueError("no JSON object in reply")
    return json.loads(raw[a:b + 1])


def llm_grade(client, model, q, sol, ans, step, attempts, max_tokens):
    pts = sol.get("points", q.get("points", 1))
    reference = sol.get("sampleAnswer") or sol.get("correctAnswer") or json.dumps(sol, ensure_ascii=False)
    prompt = PROMPT.format(points=pts, step=step, question=q.get("text", ""), solution=reference,
                           answer=ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False))
    last = None
    for i in range(1, attempts + 1):
        try:
            t = time.monotonic()
            r = client.chat.completions.create(model=model, max_tokens=max_tokens,
                                               messages=[{"role": "user", "content": prompt}])
            g = extract_json(r.choices[0].message.content)
            p = float(g["awarded_points"])
            p = min(max(round(p / step) * step, 0), pts)          # clamp and snap to the step
            return {"awarded_points": p, "explanation": str(g.get("explanation", "")),
                    "attempt": i, "seconds": round(time.monotonic() - t, 1)}
        except Exception as e:                                     # timeout, 5xx, bad JSON ...
            last = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"    attempt {i}/{attempts} failed: {last}", flush=True)
            if i < attempts:
                time.sleep(10 * i)
    raise RuntimeError(last)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exam", required=True)
    ap.add_argument("--solution", help="solution JSON; not needed if the exam JSON has an 'answer' per question")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True, help="results .jsonl (appended to; reused to resume)")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL") or DEFAULT_GRADER, help="grader for free-text")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600, help="seconds per API call")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--step", type=float, default=0.5, help="point step for free-text partial credit")
    ap.add_argument("--only", nargs="*", help="grade only these question ids (e.g. q3 q7)")
    ap.add_argument("--skip-free-text", action="store_true", help="exact-match questions only, no API calls")
    a = ap.parse_args()

    exam, answers = load_json(a.exam), load_answers(a.answers)
    solution = load_json(a.solution) if a.solution else {}
    for q in exam["questions"]:                       # Kiara's format: {id, question, options, answer}
        q.setdefault("text", q.get("question", ""))
        if not q.get("type"):
            q["type"] = "single-choice" if q.get("options") else "free-text"
        if q["id"] not in solution and "answer" in q:
            solution[q["id"]] = {"type": q["type"], "points": q.get("points", 1), "correctAnswer": q["answer"]}
    questions = [q for q in exam["questions"] if not a.only or q["id"] in a.only]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = load_done(a.out)
    print(f"{exam.get('examId', a.exam)}: {len(questions)} questions, {len(answers)} answers, "
          f"{len(done)} already graded in {a.out}", flush=True)

    client = None
    needs_llm = not a.skip_free_text and any(q.get("type") not in EXACT_TYPES and q["id"] not in done
                                             for q in questions)
    if needs_llm:
        from openai import OpenAI
        key = os.getenv("SWISSAI_RESEARCH_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            sys.exit("Export SWISSAI_RESEARCH_API_KEY (or OPENAI_API_KEY) first.")
        client = OpenAI(api_key=key, base_url=SWISS_AI, timeout=a.timeout, max_retries=0)
        print(f"Free-text grader: {a.model} (Swiss AI)", flush=True)

    failed = []
    for q in questions:
        qid, qtype = q["id"], q.get("type")
        if qid in done:
            continue
        sol = solution.get(qid)
        if sol is None:
            print(f"  {qid}: no solution entry, skipped", flush=True)
            continue
        if a.skip_free_text and qtype not in EXACT_TYPES:
            continue
        pts = sol.get("points", q.get("points", 1))
        rec = {"id": qid, "type": qtype, "max_points": pts}
        ans = answers.get(qid)
        if ans is None or (isinstance(ans, str) and not ans.strip()):
            rec.update(awarded_points=0, method="missing")
        elif qtype in EXACT_TYPES:
            rec.update(exact_grade(q, sol, ans), method="exact", answer=ans)
        elif a.skip_free_text:
            continue
        else:
            print(f"  {qid}: grading with LLM ...", flush=True)
            try:
                rec.update(llm_grade(client, a.model, q, sol, ans, a.step, a.attempts, a.max_tokens),
                           method="llm", grader=a.model)
            except RuntimeError as e:
                failed.append(qid)
                print(f"  {qid}: FAILED after {a.attempts} attempts ({e}); rerun to retry", flush=True)
                continue
        append(a.out, rec)
        done[qid] = rec
        print(f"  {qid}: {rec['awarded_points']:g}/{pts:g} ({rec['method']})", flush=True)

    graded = [done[q["id"]] for q in questions if q["id"] in done]
    got = sum(r["awarded_points"] for r in graded)
    tot = sum(r["max_points"] for r in graded)
    print(f"\nTotal: {got:g}/{tot:g} on {len(graded)}/{len(questions)} graded questions"
          + (f" ({100 * got / tot:.0f}%)" if tot else "")
          + (f"; FAILED: {', '.join(failed)} (rerun the same command to retry)" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()