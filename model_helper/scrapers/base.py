"""Base scraper class with common utilities."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from model_helper.data.models import BenchmarkInfo, BenchmarkResult, ModelInfo


class BaseScraper(ABC):
    """Base class for web scrapers."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "User-Agent": "Model-Helper/1.0 (LLM Model Research Tool)",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def fetch(self, url: str) -> httpx.Response:
        """Fetch URL with retry logic."""
        if not self.client:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")
        response = await self.client.get(url)
        response.raise_for_status()
        return response

    @abstractmethod
    async def fetch_models(self) -> list[ModelInfo]:
        """Fetch model information from source."""
        pass

    @abstractmethod
    async def fetch_benchmarks(self) -> list[BenchmarkInfo]:
        """Fetch benchmark information from source."""
        pass

    @abstractmethod
    async def fetch_results(
        self, model_ids: Optional[list[str]] = None
    ) -> list[BenchmarkResult]:
        """Fetch benchmark results from source."""
        pass


class LiteLLMScraper(BaseScraper):
    """Scraper for LiteLLM model data."""

    BASE_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

    async def fetch_models(self) -> list[ModelInfo]:
        """Fetch models from LiteLLM JSON."""
        response = await self.fetch(self.BASE_URL)
        data = response.json()

        models = []
        for model_key, model_data in data.items():
            if isinstance(model_data, dict):
                model = ModelInfo(
                    id=model_key,
                    name=model_key.replace("_", " ").title(),
                    provider=self._infer_provider(model_key, model_data),
                    family=self._infer_family(model_key),
                    context_length=model_data.get("max_input_tokens"),
                    max_output_tokens=model_data.get("max_tokens"),
                    supports_function_calling=model_data.get("supports_function_calling", False),
                    supports_vision=model_data.get("supports_vision", False),
                    input_price_per_million=model_data.get("input_cost_per_token", 0) * 1_000_000,
                    output_price_per_million=model_data.get("output_cost_per_token", 0) * 1_000_000,
                    api_model_id=model_key,
                    source="litellm",
                )
                models.append(model)

        return models

    async def fetch_benchmarks(self) -> list[BenchmarkInfo]:
        """LiteLLM doesn't have benchmark data."""
        return []

    async def fetch_results(
        self, model_ids: Optional[list[str]] = None
    ) -> list[BenchmarkResult]:
        """LiteLLM doesn't have benchmark data."""
        return []

    def _infer_provider(self, model_key: str, model_data: dict) -> str:
        """Infer provider from model key."""
        key_lower = model_key.lower()
        provider_map = {
            "gpt": "openai",
            "claude": "anthropic",
            "gemini": "google",
            "llama": "meta",
            "mistral": "mistral",
            "qwen": "alibaba",
            "gemma": "google",
            "mixtral": "mistral",
        }
        for prefix, provider in provider_map.items():
            if key_lower.startswith(prefix):
                return provider
        return model_data.get("litellm_provider", "unknown")

    def _infer_family(self, model_key: str) -> Optional[str]:
        """Infer model family from model key."""
        key_lower = model_key.lower()
        families = ["gpt-4", "gpt-3.5", "claude-3", "claude-4", "llama-3", "llama-4", "qwen-3", "qwen-2"]
        for family in families:
            if family in key_lower:
                return family
        return None


class ModelDBScraper(BaseScraper):
    """Scraper for ModelDB API."""

    BASE_URL = "https://modeldb.axiom.co/api/v1"

    async def fetch_models(self) -> list[ModelInfo]:
        """Fetch models from ModelDB API."""
        response = await self.fetch(f"{self.BASE_URL}/models?limit=500")
        data = response.json()

        models = []
        for model_data in data.get("data", []):
            model = ModelInfo(
                id=model_data.get("model_id", ""),
                name=model_data.get("model_id", "").replace("_", " ").title(),
                provider=model_data.get("provider", "unknown"),
                context_length=model_data.get("max_input_tokens"),
                max_output_tokens=model_data.get("max_tokens"),
                supports_function_calling=model_data.get("supports_function_calling", False),
                supports_vision=model_data.get("supports_vision", False),
                input_price_per_million=model_data.get("input_cost_per_million"),
                output_price_per_million=model_data.get("output_cost_per_million"),
                api_model_id=model_data.get("model_id"),
                source="modeldb",
            )
            models.append(model)

        return models

    async def fetch_benchmarks(self) -> list[BenchmarkInfo]:
        """ModelDB doesn't have benchmark info."""
        return []

    async def fetch_results(
        self, model_ids: Optional[list[str]] = None
    ) -> list[BenchmarkResult]:
        """ModelDB doesn't have benchmark results."""
        return []


class ChatbotArenaScraper(BaseScraper):
    """Scraper for Chatbot Arena (LMSYS) leaderboard."""

    LEADERBOARD_URL = "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard"

    async def fetch_models(self) -> list[ModelInfo]:
        """Fetch models from Chatbot Arena."""
        return []

    async def fetch_benchmarks(self) -> list[BenchmarkInfo]:
        """Fetch Chatbot Arena benchmark info."""
        return [
            BenchmarkInfo(
                id="chatbot-arena",
                name="Chatbot Arena",
                description="Human preference ranking of LLM chatbots through blind comparisons",
                category="human-preference",
                source_url="https://lmarena.ai",
                dataset_url="https://huggingface.co/datasets/lmsys/chatbot_arena_conversations",
            )
        ]

    async def fetch_results(
        self, model_ids: Optional[list[str]] = None
    ) -> list[BenchmarkResult]:
        """Fetch Chatbot Arena results."""
        return []


class OpenLLMLeaderboardScraper(BaseScraper):
    """Scraper for HuggingFace Open LLM Leaderboard."""

    async def fetch_models(self) -> list[ModelInfo]:
        """Fetch models from Open LLM Leaderboard."""
        return []

    async def fetch_benchmarks(self) -> list[BenchmarkInfo]:
        """Fetch benchmark info from Open LLM Leaderboard."""
        benchmarks = [
            BenchmarkInfo(
                id="mmlu",
                name="MMLU",
                description="Massive Multitask Language Understanding - 57 subjects",
                category="reasoning",
                source_url="https://huggingface.co/datasets/cais/mmlu",
                dataset_url="https://huggingface.co/datasets/cais/mmlu",
            ),
            BenchmarkInfo(
                id="hellaswag",
                name="HellaSwag",
                description="Commonsense inference for event ordering",
                category="commonsense",
                source_url="https://huggingface.co/datasets/Rowan/hellaswag",
                dataset_url="https://huggingface.co/datasets/Rowan/hellaswag",
            ),
            BenchmarkInfo(
                id="arc",
                name="ARC",
                description="AI2 Reasoning Challenge - grade-school science",
                category="reasoning",
                source_url="https://huggingface.co/datasets/ai2_arc",
                dataset_url="https://huggingface.co/datasets/ai2_arc",
            ),
            BenchmarkInfo(
                id="truthfulqa",
                name="TruthfulQA",
                description="Truthfulness evaluation against misinformation",
                category="truthfulness",
                source_url="https://huggingface.co/datasets/truthful_qa",
                dataset_url="https://huggingface.co/datasets/truthful_qa",
            ),
            BenchmarkInfo(
                id="winogrande",
                name="Winogrande",
                description="Coreference resolution at scale",
                category="reasoning",
                source_url="https://huggingface.co/datasets/allenai/winogrande",
                dataset_url="https://huggingface.co/datasets/allenai/winogrande",
            ),
            BenchmarkInfo(
                id="gsm8k",
                name="GSM8K",
                description="Grade School Math - 8.5K problems",
                category="math",
                source_url="https://huggingface.co/datasets/openai/gsm8k",
                dataset_url="https://huggingface.co/datasets/openai/gsm8k",
            ),
            BenchmarkInfo(
                id="humaneval",
                name="HumanEval",
                description="Code generation evaluation",
                category="coding",
                source_url="https://huggingface.co/datasets/openai/openai_humaneval",
                dataset_url="https://huggingface.co/datasets/openai/openai_humaneval",
            ),
        ]
        return benchmarks

    async def fetch_results(
        self, model_ids: Optional[list[str]] = None
    ) -> list[BenchmarkResult]:
        """Fetch benchmark results from Open LLM Leaderboard."""
        return []