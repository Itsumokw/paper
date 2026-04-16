from __future__ import annotations

from typing import Iterable

from openai import OpenAI

import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
SIMPLEMEM_DIR = WORKSPACE / "baseline" / "SimpleMem"
if str(SIMPLEMEM_DIR) not in sys.path:
    sys.path.insert(0, str(SIMPLEMEM_DIR))

import config  # type: ignore


DEFAULT_BASE_URLS = [
    "https://api.whatai.cc",
    "https://api.whatai.cc/v1",
    "https://api.whatai.cc/topup",
    "https://api.whatai.cc/topup/v1",
]


def short(text: object, limit: int = 240) -> str:
    s = str(text)
    return s if len(s) <= limit else s[:limit] + "..."


def try_once(base_url: str, model: str) -> None:
    print("=" * 80)
    print(f"Testing base_url: {base_url}")
    print(f"Model: {model}")
    print("-" * 80)

    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            temperature=0,
        )
        print(f"Python type: {type(response).__name__}")
        if hasattr(response, "choices"):
            try:
                content = response.choices[0].message.content
            except Exception as exc:
                print(f"choices exists but content extraction failed: {exc}")
                print(f"raw: {short(response)}")
                return
            print(f"Success. content={short(content)}")
        else:
            print("Non-standard response returned.")
            print(f"raw: {short(response)}")
    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}")


def iter_base_urls() -> Iterable[str]:
    seen = set()
    if getattr(config, "OPENAI_BASE_URL", None):
        seen.add(config.OPENAI_BASE_URL)
        yield config.OPENAI_BASE_URL
    for item in DEFAULT_BASE_URLS:
        if item not in seen:
            seen.add(item)
            yield item


def main() -> int:
    print("OpenAI-compatible relay probe")
    print("Key source: baseline/SimpleMem/config.py")
    print("This script does not print your key.")
    print()

    model = getattr(config, "LLM_MODEL", "gpt-4.1-mini")
    for base_url in iter_base_urls():
        try_once(base_url, model)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
