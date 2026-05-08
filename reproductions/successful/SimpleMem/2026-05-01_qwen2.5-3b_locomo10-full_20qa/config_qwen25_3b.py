"""
Local reproduction configuration for a single RTX 4090.

This file is ignored by git in the upstream SimpleMem repo because it may
contain private API settings. The current setup uses a local OpenAI-compatible
server backed by a local Qwen model.
"""

# ============================================================================
# LLM Configuration
# ============================================================================

OPENAI_API_KEY = "EMPTY"
OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# Hard cap for each generation. Streaming stops earlier when complete JSON
# closes, but this prevents malformed generations from running to 40960 tokens.
MAX_OUTPUT_TOKENS = 15000

# Embedding model (local, no API needed)
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024
EMBEDDING_CONTEXT_LENGTH = 32768


# ============================================================================
# Advanced LLM Features
# ============================================================================

ENABLE_THINKING = False
USE_STREAMING = True
USE_JSON_FORMAT = False


# ============================================================================
# Memory Building Parameters
# ============================================================================

WINDOW_SIZE = 40
OVERLAP_SIZE = 2


# ============================================================================
# Retrieval Parameters
# ============================================================================

SEMANTIC_TOP_K = 25
KEYWORD_TOP_K = 5
STRUCTURED_TOP_K = 5


# ============================================================================
# Database Configuration
# ============================================================================

LANCEDB_PATH = "./lancedb_data"
MEMORY_TABLE_NAME = "memory_entries"


# ============================================================================
# Parallel Processing Configuration
# ============================================================================

# Local Transformers generation shares one GPU, so start single-worker for
# reproducibility. Increase these only after smoke tests are stable.
ENABLE_PARALLEL_PROCESSING = True
MAX_PARALLEL_WORKERS = 16

ENABLE_PARALLEL_RETRIEVAL = False
MAX_RETRIEVAL_WORKERS = 16
MAX_TEST_QUESTION_WORKERS = 16

ENABLE_PLANNING = True
ENABLE_REFLECTION = True
MAX_REFLECTION_ROUNDS = 2


# ============================================================================
# LLM-as-Judge Configuration
# ============================================================================

JUDGE_API_KEY = OPENAI_API_KEY
JUDGE_BASE_URL = OPENAI_BASE_URL
JUDGE_MODEL = LLM_MODEL
JUDGE_ENABLE_THINKING = False
JUDGE_USE_STREAMING = False
JUDGE_TEMPERATURE = 0.3
