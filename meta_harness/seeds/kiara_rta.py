"""Seed: Kiara's reason-then-answer harness (solve_exam_b.py): JSON with reasoning + answer,
temperature 0.1, thinking disabled, DeepSeek. Text only (no images)."""
MODEL = "deepseek"

INSTRUCTIONS = """### Instructions
Return strictly a valid JSON object with ONLY these two keys and nothing else. Ensure all quotes and newlines inside the reasoning string are properly escaped for JSON. Do not use markdown blocks.
{
 "reasoning": "Step-by-step logical evaluation of the options.",
 "answer": "Single uppercase letter matching the correct option."
}"""


def solve(question, llm):
    options = "\n".join(f"{k}) {v}" for k, v in question["options"].items())
    prompt = f"""Solve the following multiple-choice exam question.
<question>{question['question']}
</question>
<options>{options}
</options>
{INSTRUCTIONS}"""
    reply = llm(MODEL, prompt, temperature=0.1, json_mode=True, thinking=False)
    return llm.extract_choice(reply, question["options"])
