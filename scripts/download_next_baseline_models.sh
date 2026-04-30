#!/usr/bin/env bash
set -euo pipefail

cd /home/stu0032/paper

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NLTK_DATA="${NLTK_DATA:-/home/stu0032/nltk_data}"

mkdir -p "$NLTK_DATA"

/home/stu0032/paper/.venv/bin/python - <<'PY'
import os
import nltk
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

nltk.data.path.insert(0, os.environ.get("NLTK_DATA", "/home/stu0032/nltk_data"))
for package, resource in (
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, download_dir=os.environ.get("NLTK_DATA"), quiet=True)
        nltk.data.find(resource)

llmlingua_model = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
AutoTokenizer.from_pretrained(llmlingua_model, local_files_only=True)
AutoModelForTokenClassification.from_pretrained(llmlingua_model, local_files_only=True)

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True)
print("cached next-baseline models")
PY
