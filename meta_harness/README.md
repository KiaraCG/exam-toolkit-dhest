# Meta-Harness (MC exams)

An open model (proposer, default Kimi K3.1 Pro) reads all earlier harnesses (code, scores, failed questions with
model outputs; never the answer key) and writes a new `harness.py`. Each candidate is scored by exact match on the
SEARCH exams. At the end, the best candidate is run once on held-out TEST exams.

Folder: put `meta_harness/` in `exam-toolkit-dhest/` (it imports `grade_json.py` from there).
Results go to `meta_runs/` - git-ignore it (contains model outputs).

    source .venv/bin/activate
    D=../aiscientist-dhest/data/signals-and-systems
    # 1. quick test, 5 questions per exam, 1 iteration
    python meta_harness/run_meta.py --archive meta_runs/test --search $D/2017_autumn.json --limit 5 --iterations 1
    # 2. real search (sbatch): HS17 + HS20, 10 iterations
    python meta_harness/run_meta.py --archive meta_runs/run1 --search $D/2017_autumn.json $D/2020_autumn.json --iterations 10
    # 3. held-out test, once at the end
    python meta_harness/run_meta.py --archive meta_runs/run1 --test $D/2021_autumn.json

Seeds (hand-made starting harnesses) are in `seeds/`: baseline, reason-then-answer, 3-model vote.
Settings via environment: META_PROPOSER (proposer model), META_MAX_CALLS (LLM calls per question, default 20),
META_FAKE=1 (no API calls, random answers, for testing the code).
