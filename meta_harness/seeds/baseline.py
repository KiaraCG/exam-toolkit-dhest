"""Seed: one model, final answer only."""
MODEL = "deepseek"


def format_question(q):
    opts = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
    return f"{q['question']}\n\nOptions:\n{opts}"


def solve(question, llm):
    reply = llm(MODEL, format_question(question) + "\n\nAnswer with the letter of the correct option only. "
                "End with 'Final answer: <letter>'.")
    return llm.extract_choice(reply, question["options"])
