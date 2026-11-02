import base64, os, pymupdf
from openai import OpenAI

client = OpenAI(base_url="https://api.swissai.svc.cscs.ch/v1", api_key=os.environ["OPENAI_API_KEY"])
page = pymupdf.open("demo_workspace/Rubrics/rubric.pdf")[0]
img = "data:image/png;base64," + base64.b64encode(page.get_pixmap(dpi=100).tobytes("png")).decode()

for model in ["CSCS-Inference/google/gemma-4-31B-it",
              "CSCS-Inference/swiss-ai/Apertus-v1.5-70B",
              "CSCS-Inference/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"]:
    try:
        r = client.chat.completions.create(model=model, max_tokens=100, messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img}},
            {"type": "text", "text": "What is the title of the document in this image?"}]}])
        print(model, "->", r.choices[0].message.content[:150])
    except Exception as e:
        print(model, "-> ERROR:", str(e)[:150])