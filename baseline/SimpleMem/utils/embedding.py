"""
Embedding utilities - Generate vector embeddings using SentenceTransformers
Supports Qwen3 Embedding models through SentenceTransformers interface
"""
from typing import List
import numpy as np
import config


class EmbeddingModel:
    """
    Embedding model using SentenceTransformers (supports Qwen3 and other models)
    """
    def __init__(self, model_name: str = None, use_optimization: bool = True):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_optimization = use_optimization
        self.use_api = bool(getattr(config, "EMBEDDING_USE_API", False))

        print(f"Loading embedding model: {self.model_name}")

        if self.use_api:
            self._init_openai_api_embedding()
            return

        # Check if it's a Qwen3 model (through SentenceTransformers)
        if self.model_name.startswith("qwen3"):
            self._init_qwen3_sentence_transformer()
        else:
            self._init_standard_sentence_transformer()

    def _init_openai_api_embedding(self):
        """Initialize OpenAI-compatible embedding API client."""
        from openai import OpenAI

        api_model_aliases = {
            "Qwen/Qwen3-Embedding-0.6B": "qwen3-embedding-0.6b",
            "Qwen/Qwen3-Embedding-4B": "qwen3-embedding-4b",
            "Qwen/Qwen3-Embedding-8B": "qwen3-embedding-8b",
        }
        normalized_model_name = api_model_aliases.get(self.model_name, self.model_name)
        if normalized_model_name != self.model_name:
            print(f"Normalized embedding API model name: {self.model_name} -> {normalized_model_name}")
            self.model_name = normalized_model_name

        api_key = getattr(config, "OPENAI_API_KEY", None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_USE_API=True")

        base_url = getattr(config, "OPENAI_BASE_URL", None)
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            print(f"Using embedding API base URL: {base_url}")
            print(f"Embedding requests will use: {base_url.rstrip('/')}/embeddings")

        self.client = OpenAI(**client_kwargs)
        self.dimension = int(getattr(config, "EMBEDDING_DIMENSION", 1536))
        self.model_type = "openai_api_embedding"
        self.supports_query_prompt = False
        print(f"Embedding API enabled with dimension: {self.dimension}")

    def _init_qwen3_sentence_transformer(self):
        """Initialize Qwen3 model using SentenceTransformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Map model names to actual model paths
            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B", 
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B"
            }
            
            model_path = qwen3_models.get(self.model_name, self.model_name)
            print(f"Loading Qwen3 model via SentenceTransformers: {model_path}")
            
            # Initialize with optimization settings
            if self.use_optimization:
                try:
                    # Try to use flash_attention_2 and left padding for better performance
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2", 
                            "device_map": "auto"
                        },
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True
                    )
                    print("Qwen3 loaded with flash_attention_2 optimization")
                except Exception as e:
                    print(f"Flash attention failed ({e}), using standard loading...")
                    self.model = SentenceTransformer(model_path, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_path, trust_remote_code=True)
            
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "qwen3_sentence_transformer"
            
            # Check if Qwen3 supports query prompts
            self.supports_query_prompt = hasattr(self.model, 'prompts') and 'query' in getattr(self.model, 'prompts', {})
            
            print(f"Qwen3 model loaded successfully with dimension: {self.dimension}")
            if self.supports_query_prompt:
                print("Query prompt support detected")
                
        except Exception as e:
            print(f"Failed to load Qwen3 model: {e}")
            print("Falling back to default SentenceTransformers model...")
            self._fallback_to_sentence_transformer()

    def _init_standard_sentence_transformer(self):
        """Initialize standard SentenceTransformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "sentence_transformer"
            self.supports_query_prompt = False
            print(f"SentenceTransformer model loaded with dimension: {self.dimension}")
        except Exception as e:
            print(f"Failed to load SentenceTransformer model: {e}")
            raise

    def _fallback_to_sentence_transformer(self):
        """Fallback to default SentenceTransformer model"""
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Using fallback model: {fallback_model}")
        self.model_name = fallback_model
        self._init_standard_sentence_transformer()

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode list of texts to vectors
        
        Args:
        - texts: List of texts to encode
        - is_query: Whether these are query texts (for Qwen3 prompt optimization)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # API-based embedding path
        if self.model_type == "openai_api_embedding":
            return self._encode_via_api(texts)
        
        # Use query prompt for Qwen3 models when encoding queries
        if self.model_type == "qwen3_sentence_transformer" and self.supports_query_prompt and is_query:
            return self._encode_with_query_prompt(texts)
        else:
            return self._encode_standard(texts)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode single text
        
        Args:
        - text: Text to encode
        - is_query: Whether this is a query text (for Qwen3 prompt optimization)
        """
        return self.encode([text], is_query=is_query)[0]
    
    def encode_query(self, queries: List[str]) -> np.ndarray:
        """
        Encode queries with optimal settings for Qwen3
        """
        return self.encode(queries, is_query=True)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents (no query prompt)
        """
        return self.encode(documents, is_query=False)
    
    def _encode_with_query_prompt(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Qwen3 query prompt"""
        try:
            embeddings = self.model.encode(
                texts, 
                prompt_name="query",  # Use Qwen3's query prompt
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            print(f"Query prompt encoding failed: {e}, falling back to standard encoding")
            return self._encode_standard(texts)
    
    def _encode_standard(self, texts: List[str]) -> np.ndarray:
        """Encode texts using standard method"""
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _encode_via_api(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI-compatible embedding API."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        sorted_data = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in sorted_data]
        embeddings = np.asarray(vectors, dtype=np.float32)

        # Normalize for cosine-friendly retrieval, matching local path behavior.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.divide(
            embeddings,
            norms,
            out=np.zeros_like(embeddings),
            where=norms > 0
        )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}. "
                "Please update EMBEDDING_DIMENSION in config.py to match EMBEDDING_MODEL."
            )

        return embeddings
