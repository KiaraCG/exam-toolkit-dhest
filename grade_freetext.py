#!/usr/bin/env python3
"""Grade free-text exam answers against Kiara's JSON rubric, one API call per question part.

Rubric file  (data/<course>/<exam>_freetext.json):
    questions[] -> {id, question, points, parts[] -> {id, question, points, final_answer, solution,
                    rubric[] -> {points, criterion, source}, accepted_alternatives, image_path}}
Answer file  (processed-data/<harness>/<model>/solutions_<exam>_freetext.json):
    answers[] -> {id, parts[] -> {id, final, ...any other fields, e.g. reasoning}}

- One call per part (open Swiss AI model only), up to --attempts tries each.
- The grader scores every rubric criterion; the part score is the sum, capped at the part's points.
- Each graded part is appended to a .jsonl file at once; rerunning the same command resumes.

Usage (from exam-toolkit-dhest, key exported in the shell):
    python3 grade_freetext.py \
        --rubric  ../aiscientist-dhest/data/signals-and-systems/2018_autumn_freetext.json \
        --answers ../aiscientist-dhest/processed-data/baseline/Gemma-4/solutions_2018_autumn_freetext.json \
        --out     grades/harness/baseline/2018_autumn_freetext/Gemma-4.jsonl
"""
import argparse, json, os, sys, time

SWISS_AI = "https://api.swissai.svc.cscs.ch/v1"   # rubrics/solutions may only go to open models
DEFAULT_GRADER = "CSCS-Inference/google/gemma-4-31B-it"

PROMPT = """You are a strict, fair university exam grader. Grade the candidate's answer to ONE part of an exam question.

Question context:
{context}

Part {pid} ({points} points):
{question}

Official final answer:
{final_answer}
{alternatives}
Official solution:
{solution}

Grading rubric (award each criterion independently, between 0 and its points, in steps of {step}):
{rubric}

Candidate's answer:
{answer}

Rules:
- Judge only what the candidate wrote. Give credit for correct equivalent forms and correct alternative methods.
- A criterion about the method or derivation earns points only if the candidate shows that step.
- Do not reward length or restating the question.

Return ONLY one JSON object, no markdown:
{{"criteria": [{{"criterion": 1, "awarded": <number>}}, ...one entry per rubric criterion, in order...],
  "explanation": "<one or two sentences citing what is right or wrong>"}}"""


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done[r["id"]] = r
            except (ValueError, KeyError):
                pass                                   # half-written line after a crash: regraded
    return done


def append(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def extract_json(raw):
    raw = (raw or "").strip()
    a, b = raw.find("{"), raw.rfind("}")
    if a == -1 or b <= a:
        raise ValueError("no JSON object in reply")
    s = raw[a:b + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        import re   # LaTeX like \omega in the explanation: escape stray backslashes and retry
        return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s))


def _norm(s):
    return "".join(str(s).split()).lower()


def debate_work(answer_entry, part_id, final, mode):
    """Derivation for one part from the last debate round.
    match: work of the agent(s) whose final equals the debate's final; if none matches, all agents, labelled.
    all:   all agents of the last round, labelled."""
    tr = answer_entry.get("debate_transcript") or {}
    if not tr:
        return None, "none"
    last = tr[max(tr, key=lambda r: int("".join(c for c in r if c.isdigit()) or 0))]
    agents = []
    for name, ag in last.items():
        for p in ag.get("parts", []):
            if str(p.get("id")) == str(part_id):
                agents.append((name, ag.get("model", ""), p))
    if not agents:
        return None, "none"
    hit = [x for x in agents if _norm(x[2].get("final", "")) == _norm(final)]
    if mode == "match" and hit:
        name, model, p = hit[0]
        return str(p.get("work", "")), f"match:{name}"
    text = "\n\n".join(f"[{name} ({model})]\n{p.get('work', '')}\n[{name} final] {p.get('final', '')}"
                         for name, model, p in agents)
    return text, "all"


def snap(x, step, hi):
    return min(max(round(float(x) / step) * step, 0), hi)


def answer_text(part_answer):
    """'final' plus any other text the model produced (e.g. reasoning), labelled."""
    if not isinstance(part_answer, dict):
        return str(part_answer)
    extra = {k: v for k, v in part_answer.items() if k not in ("id", "final") and v}
    text = "" if not extra else "\n".join(f"[{k}]\n{v}" for k, v in extra.items()) + "\n\n"
    return text + f"[final answer]\n{part_answer.get('final', '')}"


def grade_part(client, model, q, p, ans, step, attempts, max_tokens):
    rubric = p.get("rubric") or [{"points": p["points"], "criterion": "Correct and justified answer."}]
    alts = p.get("accepted_alternatives")
    prompt = PROMPT.format(
        context=q.get("question", ""), pid=p["id"], points=p["points"], question=p.get("question", ""),
        final_answer=p.get("final_answer", ""), solution=p.get("solution", ""), step=step,
        alternatives=f"Accepted alternatives: {json.dumps(alts, ensure_ascii=False)}\n" if alts else "",
        rubric="\n".join(f"{i}. ({c['points']} points) {c['criterion']}" for i, c in enumerate(rubric, 1)),
        answer=ans)
    last = None
    for i in range(1, attempts + 1):
        try:
            t = time.monotonic()
            r = client.chat.completions.create(model=model, max_tokens=max_tokens,
                                               messages=[{"role": "user", "content": prompt}])
            g = extract_json(r.choices[0].message.content)
            got = {int(c["criterion"]): c["awarded"] for c in g["criteria"]}
            per = [snap(got.get(k, 0), step, c["points"]) for k, c in enumerate(rubric, 1)]
            return {"awarded_points": min(sum(per), p["points"]), "criteria_points": per,
                    "explanation": str(g.get("explanation", "")), "attempt": i,
                    "seconds": round(time.monotonic() - t, 1)}
        except Exception as e:                       # timeout, 5xx, bad JSON ...
            last = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"    attempt {i}/{attempts} failed: {last}", flush=True)
            if i < attempts:
                time.sleep(10 * i)
    raise RuntimeError(last)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True, help="results .jsonl (appended to; reused to resume)")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL") or DEFAULT_GRADER)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600, help="seconds per API call")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--final-only", action="store_true", help="grade only the 'final' field (ignore work/reasoning)")
    ap.add_argument("--debate-work", choices=["match", "all"],
                    help="debate files: take the derivation from the last round of debate_transcript "
                         "(match = agent whose final equals the debate final, falls back to all agents)")
    ap.add_argument("--only", nargs="*", help="grade only these ids, e.g. 1/a 7/b")
    a = ap.parse_args()

    rubric, answers = load(a.rubric), load(a.answers)
    given, sources = {}, {}
    for x in answers.get("answers", []):
        for p in x.get("parts", []):
            key = (str(x["id"]), str(p["id"]))
            p = dict(p)
            if a.debate_work and not str(p.get("work", "")).strip():
                w, src = debate_work(x, p["id"], p.get("final", ""), a.debate_work)
                if w:
                    p["work"] = w
                sources[key] = src
            given[key] = p
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = load_done(a.out)
    parts = [(q, p) for q in rubric["questions"] for p in q.get("parts", [])]
    if a.only:
        parts = [(q, p) for q, p in parts if f"{q['id']}/{p['id']}" in a.only]
    print(f"{rubric.get('examId', a.rubric)}: {len(parts)} parts, {len(given)} answered, "
          f"{len(done)} already graded; grader {a.model}", flush=True)

    client = None
    failed = []
    for q, p in parts:
        pid = f"{q['id']}/{p['id']}"
        if pid in done:
            continue
        rec = {"id": pid, "type": "free-text", "max_points": p["points"]}
        ans = given.get((str(q["id"]), str(p["id"])))
        if ans is None or not str(ans.get("final", "") if isinstance(ans, dict) else ans).strip():
            rec.update(awarded_points=0, method="missing")
        else:
            if client is None:
                from openai import OpenAI
                key = os.getenv("SWISSAI_RESEARCH_API_KEY") or os.getenv("OPENAI_API_KEY")
                if not key:
                    sys.exit("Export SWISSAI_RESEARCH_API_KEY first.")
                client = OpenAI(api_key=key, base_url=SWISS_AI, timeout=a.timeout, max_retries=0)
            print(f"  {pid}: grading ...", flush=True)
            try:
                rec.update(grade_part(client, a.model, q, p,
                                      str(ans.get("final", "")) if a.final_only and isinstance(ans, dict) else answer_text(ans),
                                      a.step, a.attempts, a.max_tokens),
                           method="llm", grader=a.model)
            except RuntimeError as e:
                failed.append(pid)
                print(f"  {pid}: FAILED after {a.attempts} attempts ({e}); rerun to retry", flush=True)
                continue
        if (str(q["id"]), str(p["id"])) in sources:
            rec["work_source"] = sources[(str(q["id"]), str(p["id"]))]
        append(a.out, rec)
        done[pid] = rec
        print(f"  {pid}: {rec['awarded_points']:g}/{p['points']:g} ({rec['method']})", flush=True)

    graded = [done[f"{q['id']}/{p['id']}"] for q, p in parts if f"{q['id']}/{p['id']}" in done]
    got, tot = sum(r["awarded_points"] for r in graded), sum(r["max_points"] for r in graded)
    print(f"\nTotal: {got:g}/{tot:g} on {len(graded)}/{len(parts)} graded parts"
          + (f" ({100 * got / tot:.0f}%)" if tot else "")
          + (f"; FAILED: {', '.join(failed)} (rerun the same command to retry)" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()