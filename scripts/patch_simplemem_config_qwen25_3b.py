"""Patch SimpleMem config.py to use local vLLM Qwen2.5-3B-Instruct."""
from pathlib import Path
import re
import sys

CONF = Path("/home/stu0032/paper/baseline/SimpleMem/config.py")

def patch():
    s = CONF.read_text()
    s = re.sub(r'^OPENAI_API_KEY = .*$', 'OPENAI_API_KEY = "EMPTY"', s, flags=re.M)
    s = re.sub(r'^OPENAI_BASE_URL = .*$', 'OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"', s, flags=re.M)
    s = re.sub(r'^LLM_MODEL = .*$', 'LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"', s, flags=re.M)
    CONF.write_text(s)
    print("Config patched for Qwen2.5-3B-Instruct + local vLLM")

def restore(src: Path):
    import shutil
    shutil.copy2(src, CONF)
    print("Config restored")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore(Path(sys.argv[2]))
    else:
        patch()
