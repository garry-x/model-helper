"""Base scraper class with common utilities."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

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
        # Handle both list response and dict with "data" key
        model_list = data if isinstance(data, list) else data.get("data", [])
        for model_data in model_list:
            model = ModelInfo(
                id=model_data.get("model_id", ""),
                name=model_data.get("model_id", "").replace("_", " ").title(),
                provider=model_data.get("provider_id") or model_data.get("provider_name") or "unknown",
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


class HuggingFaceScraper(BaseScraper):
    """Scraper for Hugging Face Hub model data.

    Tries the primary URL first, then falls back to mirrors on network errors.
    Mirrors are configured in config.json (hf_mirrors).
    """

    PRIMARY_URL = "https://huggingface.co/api/models"
    MAX_PAGES = 5
    PAGE_LIMIT = 100

    def __init__(self, provider_authors: dict[str, list[str]] | None = None,
                 mirrors: list[str] | None = None, timeout: int = 30,
                 on_progress: Optional[Callable] = None):
        super().__init__(timeout=timeout)
        self.provider_authors = provider_authors or {}
        self.on_progress = on_progress
        # Build URL list: primary first, then each mirror + /api/models
        self.base_urls = [self.PRIMARY_URL]
        for m in (mirrors or []):
            m = m.rstrip("/")
            self.base_urls.append(f"{m}/api/models")

    async def fetch_models(self) -> list[ModelInfo]:
        """Fetch models from Hugging Face Hub.

        If provider_authors is set, searches by those specific authors.
        Otherwise fetches the most-downloaded text-generation models.
        """
        models: list[ModelInfo] = []

        if self.provider_authors:
            # Search by specific authors for each configured provider
            all_authors: set[str] = set()
            provider_for_author: dict[str, str] = {}
            for provider, authors in self.provider_authors.items():
                for author in authors:
                    all_authors.add(author.lower())
                    provider_for_author[author.lower()] = provider.lower()

            for author in all_authors:
                page_models = await self._fetch_by_author(author)
                for m in page_models:
                    # Normalize provider from author mapping
                    m.provider = provider_for_author.get(author, m.provider)
                    models.append(m)
        else:
            # Fetch top downloaded text-generation models
            models = await self._fetch_top_models()

        return models

    async def _fetch_by_author(self, author: str) -> list[ModelInfo]:
        """Fetch models by a specific HF author (org/user)."""
        models: list[ModelInfo] = []
        cursor = None
        pages = 0

        while pages < self.MAX_PAGES:
            params: dict[str, str | int] = {
                "author": author,
                "filter": "text-generation",
                "sort": "downloads",
                "direction": "-1",
                "limit": self.PAGE_LIMIT,
                "full": "true",
            }
            data, next_cursor = await self._fetch_page(params, cursor)
            page_count = 0
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    models.append(self._parse_model(item))
                    page_count += 1

            pages += 1
            if self.on_progress:
                self.on_progress(author, pages, page_count)
            if not next_cursor or len(data) < self.PAGE_LIMIT:
                break
            cursor = next_cursor

        return models

    async def _fetch_top_models(self) -> list[ModelInfo]:
        """Fetch the most-downloaded text-generation models."""
        models: list[ModelInfo] = []
        cursor = None
        pages = 0

        while pages < self.MAX_PAGES:
            params: dict[str, str | int] = {
                "filter": "text-generation",
                "sort": "downloads",
                "direction": "-1",
                "limit": self.PAGE_LIMIT,
                "full": "true",
            }
            data, next_cursor = await self._fetch_page(params, cursor)
            page_count = 0
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    models.append(self._parse_model(item))
                    page_count += 1

            pages += 1
            if self.on_progress:
                self.on_progress("", pages, page_count)
            if not next_cursor or len(data) < self.PAGE_LIMIT:
                break
            cursor = next_cursor

        return models

    async def _fetch_page(self, params: dict, cursor: str | None = None,
                          _tried_urls: set[str] | None = None) -> tuple[list, str | None]:
        """Fetch one page, falling back to mirrors on network errors.

        Tries each base URL in order. On httpx.ConnectError / TimeoutException
        tries the next mirror. HTTP errors (4xx/5xx) are raised immediately.
        """
        import httpx

        if _tried_urls is None:
            _tried_urls = set()

        query = "&".join(f"{k}={v}" for k, v in params.items())
        last_error: Exception | None = None

        for i, base_url in enumerate(self.base_urls):
            if base_url in _tried_urls:
                continue
            url = f"{base_url}?{query}"
            tag = "primary" if i == 0 else f"mirror [{base_url}]"

            try:
                response = await self.fetch(url)
                data = response.json()
                if not isinstance(data, list):
                    data = []
                next_cursor = None
                link = response.headers.get("link") or response.headers.get("Link")
                if link:
                    import re
                    match = re.search(r'cursor=([^&>]+)', link)
                    if match:
                        next_cursor = match.group(1)
                # Success — remember this URL for subsequent pages
                if i > 0:
                    self.base_urls.insert(0, self.base_urls.pop(i))
                return data, next_cursor

            except (httpx.NetworkError, httpx.TimeoutException,
                    httpx.RemoteProtocolError, OSError) as e:
                last_error = e
                _tried_urls.add(base_url)
                continue  # try next mirror

        # All URLs failed
        raise RuntimeError(
            f"Failed to reach HF API after trying {len(self.base_urls)} URL(s). "
            f"Last error: {last_error}"
        )

    def _parse_model(self, item: dict) -> ModelInfo:
        """Map a Hugging Face model dict to ModelInfo."""
        model_id = item.get("id", "")
        author = item.get("author", "unknown")
        config = item.get("config") or {}

        # Total parameters from safetensors (largest precision)
        total_params = None
        st = item.get("safetensors") or {}
        st_params = st.get("parameters") or {}
        if st_params:
            # st_params is dict like {"BF16": 8e9, "F32": 16e9}
            total_params = int(max(st_params.values()))

        # Architecture
        arch = None
        raw_archs = config.get("architectures") or []
        if raw_archs:
            arch_name = raw_archs[0].lower()
            if "moe" in arch_name or "mixtral" in arch_name:
                arch = "moe"
            elif "dense" in arch_name:
                arch = "dense"

        # Context length
        ctx = config.get("max_position_embeddings")

        # Capabilities from tags
        tags = [t.lower() for t in item.get("tags", [])]
        supports_vision = any(t in tags for t in ("image-to-text", "image-text-to-text", "visual", "vision"))
        supports_function_calling = any(t in tags for t in ("function-calling", "tool-use", "tool_calling"))

        card = item.get("cardData") or {}

        # Derive family from tags or model_name
        family = None
        model_type = config.get("model_type", "").lower()
        if model_type:
            family = model_type

        # Name: take the last segment of the model_id
        name = model_id.split("/")[-1] if "/" in model_id else model_id

        return ModelInfo(
            id=model_id,
            model_id=model_id,
            name=name,
            provider=author,
            family=family,
            total_params=total_params,
            architecture=arch,
            context_length=ctx,
            supports_vision=supports_vision,
            supports_function_calling=supports_function_calling,
            license=str(card.get("license", "")) if card.get("license") else None,
            huggingface_id=model_id,
            source="huggingface",
        )

    async def fetch_benchmarks(self) -> list[BenchmarkInfo]:
        return []

    async def fetch_results(self, model_ids: list[str] | None = None) -> list[BenchmarkResult]:
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