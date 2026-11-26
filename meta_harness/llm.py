"""LLM access for harnesses: Swiss AI only (open models), retries, rate-limit waits, per-question traces."""
import os, random, re, threading, time

SWISS_AI = "https://api.swissai.svc.cscs.ch/v1"
MODELS = {
    "gemma": "CSCS-Inference/google/gemma-4-31B-it",
    "qwen": "SwissAI-Research/Qwen/Qwen3.8-27B",
    "deepseek": "RCP-AIaaS/deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "apertus": "CSCS-Inference/swiss-ai/Apertus-v1.5-70B",
}
_local = threading.local()
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.getenv("SWISSAI_RESEARCH_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Export SWISSAI_RESEARCH_API_KEY first.")
        _client = OpenAI(api_key=key, base_url=SWISS_AI, timeout=float(os.getenv("META_TIMEOUT", 600)), max_retries=0)
    return _client


def start_trace():
    _local.trace, _local.calls = [], 0


def get_trace():
    return getattr(_local, "trace", []), getattr(_local, "calls", 0)


class LLM:
    """Passed to harness.solve(question, llm). Call llm(model, prompt) -> reply text."""
    models = MODELS
    max_calls_per_question = int(os.getenv("META_MAX_CALLS", 20))

    def __call__(self, model, prompt, max_tokens=4000, temperature=None, system=None, attempts=3):
        model = MODELS.get(model, model)
        if not any(model == m for m in MODELS.values()):
            raise ValueError(f"model not allowed: {model} (use one of {list(MODELS)})")
        _local.calls = getattr(_local, "calls", 0) + 1
        if _local.calls > self.max_calls_per_question:
            raise RuntimeError(f"more than {self.max_calls_per_question} LLM calls for one question")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        if os.getenv("META_FAKE"):
            reply = f"Reasoning... Final answer: {random.choice('ABCD')}"
        else:
            reply, last = None, None
            for i in range(attempts):
                try:
                    kw = {"temperature": temperature} if temperature is not None else {}
                    r = _get_client().chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, **kw)
                    reply = r.choices[0].message.content or ""
                    break
                except Exception as e:
                    last = e
                    time.sleep(60 if "429" in str(e) else 10 * (i + 1))
            if reply is None:
                raise RuntimeError(f"{model} failed: {type(last).__name__}: {str(last)[:150]}")
        getattr(_local, "trace", []).append({"model": model, "prompt": prompt[-1500:], "reply": reply[-2500:]})
        return reply

    @staticmethod
    def extract_choice(text, options):
        """Last option letter mentioned as the answer in a reply (e.g. 'Final answer: C'); '' if none."""
        letters = [k.upper() for k in options]
        pats = [r"final answer\s*[:=]?\s*\(?([A-Z])\b", r"answer\s*[:=]\s*\(?([A-Z])\b", r"\b([A-Z])\)?\s*$"]
        for pat in pats:
            hits = [h.upper() for h in re.findall(pat, text or "", flags=re.I | re.M) if h.upper() in letters]
            if hits:
                return hits[-1]
        return ""
