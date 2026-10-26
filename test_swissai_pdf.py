import base64, os, pymupdf
from openai import OpenAI

client = OpenAI(base_url="https://api.swissai.svc.cscs.ch/v1",
                api_key=os.environ["SWISSAI_RESEARCH_API_KEY"])
MODEL = "CSCS-Inference/google/gemma-4-31B-it"
doc = pymupdf.open("demo_workspace/Rubrics/rubric.pdf")
images = [{"type": "input_image",
           "image_url": "data:image/png;base64," + base64.b64encode(p.get_pixmap(dpi=100).tobytes("png")).decode()}
          for p in doc]
QUESTION = {"type": "input_text", "text": "What is the title of this document and how many criteria does it list?"}
SCHEMA = {"format": {"type": "json_schema", "name": "answer", "strict": True,
          "schema": {"type": "object", "additionalProperties": False, "required": ["title", "criteria"],
                     "properties": {"title": {"type": "string"}, "criteria": {"type": "integer"}}}}}

tests = {
    "1. images, no JSON":   dict(input=[{"role": "user", "content": images + [QUESTION]}]),
    "2. text only, JSON":   dict(input="Return the title 'Test' and criteria 5.", text=SCHEMA),
    "3. images + JSON":     dict(input=[{"role": "user", "content": images + [QUESTION]}], text=SCHEMA),
}
for name, kwargs in tests.items():
    try:
        r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": images[0]["image_url"]}},
            {"type": "text", "text": "What is the title of this document?"}]}])
        print("4. images via chat API ->", r.choices[0].message.content[:300])
    except Exception as e:
        print("4. images via chat API -> ERROR:", str(e)[:300])
        