"""Seed: one model reasons step by step, then gives the letter."""
MODEL = "deepseek"


def format_question(q):
    opts = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
    return f"{q['question']}\n\nOptions:\n{opts}"


def solve(question, llm):
    reply = llm(MODEL, format_question(question) + "\n\nThink step by step, check your result, then end with "
                "'Final answer: <letter>'.", max_tokens=4000)
    return llm.extract_choice(reply, question["options"])
