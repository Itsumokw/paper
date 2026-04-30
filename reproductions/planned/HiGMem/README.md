# Planned Reproduction: HiGMem

Paper:

- HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents
- Code: `https://github.com/ZeroLoss-Lab/HiGMem`
- Local checkout: `/home/stu0032/paper/baseline/HiGMem`
- Local commit: `f275072`

Planned setup:

- Dataset: `/home/stu0032/paper/baseline/HiGMem/data/locomo10.json`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Model path: `/home/stu0032/paper/models/Qwen2.5-3B-Instruct-clean`
- vLLM base URL: `http://127.0.0.1:8000/v1`

Run guides and scripts:

- Guide: `/home/stu0032/paper/docs/higmem_reproduction_plan.md`
- Start vLLM: `/home/stu0032/paper/start_vllm_qwen25_3b.sh`
- Smoke run: `/home/stu0032/paper/scripts/run_higmem_qwen25_3b_smoke.sh`
- Full run: `/home/stu0032/paper/scripts/run_higmem_qwen25_3b_full.sh`

Status:

- Code cloned.
- Main Python files compile.
- LoCoMo10 loader works.
- Current `.venv` has required OpenAI-compatible backend dependencies.
- `ollama` is missing, but it is not needed for the planned vLLM backend.

