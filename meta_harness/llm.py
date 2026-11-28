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

    def __call__(self, model, prompt, max_tokens=None, temperature=None, system=None, attempts=3,
                 json_mode=False, thinking=None):
        max_tokens = max_tokens or int(os.getenv("META_MAX_TOKENS", 16000))   # reasoning models need room to think
        model = MODELS.get(model, model)
        if not any(model == m for m in MODELS.values()):
            raise ValueError(f"model not allowed: {model} (use one of {list(MODELS)})")
        _local.calls = getattr(_local, "calls", 0) + 1
        if _local.calls > self.max_calls_per_question:
            raise RuntimeError(f"more than {self.max_calls_per_question} LLM calls for one question")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        if os.getenv("META_FAKE"):
            letter = random.choice("ABCD")
            reply = (f'{{"reasoning": "...", "answer": "{letter}"}}' if json_mode else f"Reasoning... Final answer: {letter}")
        else:
            reply, last = None, None
            for i in range(attempts):
                try:
                    kw = {"temperature": temperature} if temperature is not None else {}
                    if json_mode:
                        kw["response_format"] = {"type": "json_object"}
                    if thinking is not None:
                        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
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
        m = re.search(r'"answer"\s*:\s*"\(?([A-Za-z])\)?"', text or "")      # JSON replies
        if m and m.group(1).upper() in letters:
            return m.group(1).upper()
        pats = [r"final answer\s*[:=]?\s*\(?([A-Z])\b", r"answer\s*[:=]\s*\(?([A-Z])\b", r"\b([A-Z])\)?\s*$"]
        for pat in pats:
            hits = [h.upper() for h in re.findall(pat, text or "", flags=re.I | re.M) if h.upper() in letters]
            if hits:
                return hits[-1]
        return ""
