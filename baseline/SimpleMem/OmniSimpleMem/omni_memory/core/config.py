"""Configuration objects for Omni-SimpleMem.

The upstream checkout referenced these classes from many modules but did not
include the module.  This file supplies the minimal configuration surface used
by the LoCoMo benchmark runner and the text-only memory pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StorageConfig:
    base_dir: str = "./omni_memory_data"
    cold_storage_dir: str = "./omni_memory_data/cold_storage"
    index_dir: str = "./omni_memory_data/index"
    use_s3: bool = False
    s3_bucket: str = ""
    s3_prefix: str = ""
    organize_by_date: bool = True
    organize_by_modality: bool = True
    auto_cleanup_enabled: bool = False


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 32
    api_key: str | None = None
    visual_embedding_model: str = "openai/clip-vit-base-patch32"
    visual_embedding_dim: int = 512
    use_proxy_for_tongyi: bool = False
    tongyi_proxy_url: str | None = None


@dataclass
class LLMConfig:
    api_key: str | None = None
    api_base_url: str | None = None
    summary_model: str = "gpt-4o"
    query_model: str = "gpt-4o"
    caption_model: str = "gpt-4o"
    whisper_model: str = "whisper-1"
    temperature: float = 0.0
    max_tokens: int = 1000


@dataclass
class RetrievalConfig:
    default_top_k: int = 20
    token_budget: int = 4096
    max_expanded_items: int = 8
    auto_expand_threshold: float = 0.55
    enable_hybrid_search: bool = True
    enable_graph_traversal: bool = True
    enable_multi_query_retrieval: bool = False


@dataclass
class EntropyTriggerConfig:
    visual_encoder: str = "none"
    visual_model_name: str = "openai/clip-vit-base-patch32"
    visual_similarity_threshold_high: float = 0.95
    visual_similarity_threshold_low: float = 0.75
    enable_visual_trigger: bool = False
    audio_energy_threshold: float = 0.01
    audio_vad_threshold: float = 0.5
    audio_min_speech_duration_ms: int = 250
    enable_audio_trigger: bool = False


@dataclass
class EventConfig:
    event_time_window_seconds: int = 1800
    summarize_on_close: bool = False
    max_maus_for_summary: int = 20


@dataclass
class RouterConfig:
    router_mode: str = "heuristic"
    gini_threshold: float = 0.4
    top1_threshold: float = 0.7
    gap_threshold: float = 0.15
    episodic_margin: float = 0.1
    close_margin: float = 0.05
    shadow_mode: bool = True
    benchmark_safe: bool = True


@dataclass
class OmniMemoryConfig:
    storage: StorageConfig = field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    entropy_trigger: EntropyTriggerConfig = field(default_factory=EntropyTriggerConfig)
    event: EventConfig = field(default_factory=EventConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    enable_self_evolution: bool = False
    benchmark_safe: bool = True

    @classmethod
    def create_default(cls) -> "OmniMemoryConfig":
        return cls()

    def ensure_directories(self) -> None:
        Path(self.storage.base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.storage.cold_storage_dir).mkdir(parents=True, exist_ok=True)
        Path(self.storage.index_dir).mkdir(parents=True, exist_ok=True)
