"""Quick preflight check that vLLM is responding."""
import sys
from openai import OpenAI

c = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
r = c.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "ok"}],
    max_tokens=4,
)
print("vLLM preflight OK:", r.choices[0].message.content)
