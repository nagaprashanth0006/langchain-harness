"Centralized configuration for the RAG system."

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedderType(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    OLLAMA = "ollama"


class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    IN_MEMORY = "in_memory"


class ChunkerType(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    SENTENCE = "sentence"


class RetrieverType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class RerankerType(str, Enum):
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    LLM = "llm"
    NONE = "none"


class GeneratorType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    cohere_api_key: str = Field(default="", description="Cohere API key")

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama daemon URL. Change if running in Docker or remote host.",
    )
    ollama_chat_model: str = Field(
        default="llama3.2",
        description=(
            "Chat/generation model pulled in Ollama. "
            "Alternatives: llama3.1, mistral, mixtral, gemma2, qwen2.5, phi3, deepseek-r1"
        ),
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        description=(
            "Embedding model pulled in Ollama. "
            "Alternatives: mxbai-embed-large (dim=1024), all-minilm (dim=384), bge-m3 (dim=1024)"
        ),
    )
    ollama_timeout: int = Field(
        default=120,
        description="Request timeout in seconds. Increase for large models on slow hardware.",
    )
    ollama_keep_alive: str = Field(
        default="5m",
        description="How long Ollama keeps the model in VRAM after last request. '0' = unload immediately.",
    )

    chunker_type: ChunkerType = Field(default=ChunkerType.RECURSIVE)
    chunk_size: int = Field(default=512, description="Target chunk size in tokens")
    chunk_overlap: int = Field(
        default=64, description="Overlap between consecutive chunks"
    )
    max_file_size_mb: int = Field(default=50)

    embedder_type: EmbedderType = Field(default=EmbedderType.OLLAMA)
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    huggingface_embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")

    cohere_embedding_model: str = Field(default="embed-english-v3.0")
    embedding_batch_size: int = Field(default=32)
    embedding_cache_dir: Path = Field(default=Path(".cache/embeddings"))
    embedding_dim: int = Field(
        default=768,
        description=(
            "Must match the chosen model's output dimension. "
            "nomic-embed-text=768 | mxbai-embed-large=1024 | all-minilm=384 | "
            "bge-m3=1024 | text-embedding-3-small=1536 | bge-large=1024"
        ),
    )

    vector_store_type: VectorStoreType = Field(default=VectorStoreType.CHROMA)
    chroma_persist_dir: Path = Field(default=Path(".data/chroma"))
    chroma_collection_name: str = Field(default="rag_documents")
    pgvector_connection_string: str = Field(
        default="postgresql+psycopg://user:pass@localhost:5432/ragdb"
    )
    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = Field(default="rag-index")

    retriever_type: RetrieverType = Field(default=RetrieverType.HYBRID)
    top_k_retrieval: int = Field(
        default=20, description="Candidates fetched before reranking"
    )
    top_k_final: int = Field(
        default=5, description="Docs sent to generation after reranking"
    )
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0.0=pure BM25, 1.0=pure dense; 0.5 balanced hybrid",
    )

    reranker_type: RerankerType = Field(default=RerankerType.CROSS_ENCODER)
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    cohere_reranker_model: str = Field(default="rerank-english-v3.0")

    generator_type: GeneratorType = Field(default=GeneratorType.OLLAMA)
    anthropic_model: str = Field(default="claude-sonnet-4-6")
    openai_model: str = Field(default="gpt-4o")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.1, description="Low temp for factual RAG")
    stream: bool = Field(default=False)

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    cors_origins: list[str] = Field(default=["*"])

    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True)


settings = Settings()
