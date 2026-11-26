"""Seed: three models reason independently; majority vote, ties broken by DeepSeek."""
from collections import Counter

MODELS = ["deepseek", "qwen", "gemma"]


def format_question(q):
    opts = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
    return f"{q['question']}\n\nOptions:\n{opts}"


def solve(question, llm):
    prompt = format_question(question) + "\n\nThink step by step, check your result, then end with 'Final answer: <letter>'."
    answers = [llm.extract_choice(llm(m, prompt), question["options"]) for m in MODELS]
    votes = Counter(a for a in answers if a)
    if not votes:
        return ""
    best, n = votes.most_common(1)[0]
    return best if n > 1 else answers[0]
