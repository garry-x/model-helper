"""Data models for model information and benchmarks."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ArchitectureType(str, Enum):
    """Model architecture types."""

    DENSE = "dense"
    MOE = "moe"
    SPARSE_MOE = "sparse_moe"
    MHA = "mha"
    GQA = "gqa"


class Modality(str, Enum):
    """Supported modalities."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class ModelInfo(BaseModel):
    """Comprehensive model information."""

    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Display name")
    provider: str = Field(..., description="Provider name (e.g., openai, anthropic)")
    family: Optional[str] = Field(None, description="Model family (e.g., gpt-4, claude-4)")

    total_params: Optional[int] = Field(None, description="Total parameters")
    activated_params: Optional[int] = Field(None, description="Activated parameters (for MoE)")
    architecture: Optional[ArchitectureType] = Field(None, description="Architecture type")
    num_experts: Optional[int] = Field(None, description="Number of experts (for MoE)")
    attention_type: Optional[str] = Field(None, description="Attention type (MHA, GQA, MQA)")

    context_length: Optional[int] = Field(None, description="Maximum context length (tokens)")
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens")

    input_modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    output_modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])

    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = True

    input_price_per_million: Optional[float] = Field(None, description="Input price per 1M tokens")
    output_price_per_million: Optional[float] = Field(None, description="Output price per 1M tokens")

    release_date: Optional[datetime] = None
    knowledge_cutoff: Optional[datetime] = None
    training_tokens: Optional[int] = Field(None, description="Training tokens (e.g., 15T)")
    license: Optional[str] = None

    huggingface_id: Optional[str] = Field(None, description="HuggingFace model ID")
    api_model_id: Optional[str] = Field(None, description="Provider API model ID")

    cached_at: datetime = Field(default_factory=datetime.now)
    source: str = Field(default="manual", description="Data source")

    model_config = {"extra": "allow"}


class BenchmarkInfo(BaseModel):
    """Benchmark information."""

    id: str = Field(..., description="Benchmark ID")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Benchmark description")
    category: str = Field(..., description="Category (reasoning, coding, math, etc.)")
    source_url: str = Field(..., description="Official website URL")
    dataset_url: Optional[str] = Field(None, description="Dataset URL")

    model_config = {"extra": "allow"}


class BenchmarkResult(BaseModel):
    """Benchmark result for a model."""

    id: Optional[int] = None
    model_id: str = Field(..., description="Model ID")
    benchmark_id: str = Field(..., description="Benchmark ID")

    score: Optional[float] = Field(None, description="Score (0-100)")
    score_formatted: Optional[str] = Field(None, description="Formatted score (e.g., '85.2%')")

    rank: Optional[int] = Field(None, description="Rank on leaderboard")
    elo_rating: Optional[float] = Field(None, description="Elo rating (for Chatbot Arena)")

    evaluation_method: Optional[str] = Field(None, description="How the score was obtained")
    evaluated_at: Optional[datetime] = Field(None, description="Evaluation timestamp")

    source: str = Field(default="scraped", description="Data source")

    model_config = {"extra": "allow"}


class CacheStatus(BaseModel):
    """Cache status information."""

    total_models: int = 0
    total_benchmarks: int = 0
    total_results: int = 0

    last_model_update: Optional[datetime] = None
    last_benchmark_update: Optional[datetime] = None

    cache_size_mb: Optional[float] = None


class SearchResult(BaseModel):
    """Search result with relevance score."""

    model: ModelInfo
    score: float = Field(..., description="Similarity score (0-100)")
    matched_on: str = Field(..., description="Field that matched")


class BenchmarkWithResults(BenchmarkInfo):
    """Benchmark with associated results."""

    results: list[BenchmarkResult] = Field(default_factory=list)


class ModelWithBenchmarks(ModelInfo):
    """Model with benchmark results."""

    benchmarks: list[BenchmarkResult] = Field(default_factory=list)