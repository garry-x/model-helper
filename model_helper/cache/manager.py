"""Cache manager for SQLite operations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from model_helper.data.models import (
    ArchitectureType,
    BenchmarkInfo,
    BenchmarkResult,
    CacheStatus,
    ModelInfo,
)
from model_helper.data.schema import DB_PATH, init_db


class CacheManager:
    """Manages SQLite cache for models and benchmarks."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH

    async def init(self):
        """Initialize the database."""
        await init_db(self.db_path)

    async def save_model(self, model: ModelInfo) -> None:
        """Save or update a model."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO models (
                    id, model_id, name, provider, family, total_params, activated_params,
                    architecture, num_experts, attention_type, context_length,
                    max_output_tokens, input_modalities, output_modalities,
                    supports_function_calling, supports_vision, supports_reasoning,
                    supports_json_mode, supports_streaming, input_price_per_million,
                    output_price_per_million, release_date, knowledge_cutoff,
                    training_tokens, license, huggingface_id, api_model_id,
                    cached_at, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model.id,
                    model.model_id,
                    model.name,
                    model.provider,
                    model.family,
                    model.total_params,
                    model.activated_params,
                    model.architecture.value if model.architecture else None,
                    model.num_experts,
                    model.attention_type,
                    model.context_length,
                    model.max_output_tokens,
                    json.dumps([m.value for m in model.input_modalities]),
                    json.dumps([m.value for m in model.output_modalities]),
                    int(model.supports_function_calling),
                    int(model.supports_vision),
                    int(model.supports_reasoning),
                    int(model.supports_json_mode),
                    int(model.supports_streaming),
                    model.input_price_per_million,
                    model.output_price_per_million,
                    model.release_date.isoformat() if model.release_date else None,
                    model.knowledge_cutoff.isoformat() if model.knowledge_cutoff else None,
                    model.training_tokens,
                    model.license,
                    model.huggingface_id,
                    model.api_model_id,
                    model.cached_at.isoformat(),
                    model.source,
                ),
            )
            await db.commit()

    async def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a model by ID or name (fuzzy search)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM models WHERE id = ?", (model_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_model(row)
            async with db.execute(
                "SELECT * FROM models WHERE name LIKE ? LIMIT 1", (f"%{model_id}%",)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_model(row)
                return None

    async def list_models(
        self,
        provider: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelInfo]:
        """List models with optional filtering."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            query = "SELECT * FROM models"
            params = []
            if provider:
                query += " WHERE provider = ?"
                params.append(provider)
            query += " ORDER BY name LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_model(row) for row in rows]

    async def save_benchmark(self, benchmark: BenchmarkInfo) -> None:
        """Save or update a benchmark."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO benchmarks (id, name, description, category, source_url, dataset_url)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    benchmark.id,
                    benchmark.name,
                    benchmark.description,
                    benchmark.category,
                    benchmark.source_url,
                    benchmark.dataset_url,
                ),
            )
            await db.commit()

    async def get_benchmark(self, benchmark_id: str) -> Optional[BenchmarkInfo]:
        """Get a benchmark by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM benchmarks WHERE id = ?", (benchmark_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_benchmark(row)
                return None

    async def list_benchmarks(self, category: Optional[str] = None) -> list[BenchmarkInfo]:
        """List all benchmarks."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            query = "SELECT * FROM benchmarks"
            if category:
                query += " WHERE category = ?"
                async with db.execute(query, (category,)) as cursor:
                    rows = await cursor.fetchall()
            else:
                async with db.execute(query) as cursor:
                    rows = await cursor.fetchall()
            return [self._row_to_benchmark(row) for row in rows]

    async def save_result(self, result: BenchmarkResult) -> None:
        """Save or update a benchmark result."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO benchmark_results (
                    model_id, benchmark_id, score, score_formatted, rank,
                    elo_rating, evaluation_method, evaluated_at, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.model_id,
                    result.benchmark_id,
                    result.score,
                    result.score_formatted,
                    result.rank,
                    result.elo_rating,
                    result.evaluation_method,
                    result.evaluated_at.isoformat() if result.evaluated_at else None,
                    result.source,
                ),
            )
            await db.commit()

    async def get_results_for_model(self, model_id: str) -> list[BenchmarkResult]:
        """Get all benchmark results for a model."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM benchmark_results WHERE model_id = ?", (model_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_result(row) for row in rows]

    async def get_results_for_benchmark(
        self, benchmark_id: str, limit: int = 50
    ) -> list[BenchmarkResult]:
        """Get all model results for a benchmark."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM benchmark_results WHERE benchmark_id = ? ORDER BY score DESC LIMIT ?",
                (benchmark_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_result(row) for row in rows]

    async def get_status(self) -> CacheStatus:
        """Get cache status."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM models") as cursor:
                total_models = (await cursor.fetchone())[0]

            async with db.execute("SELECT COUNT(*) FROM benchmarks") as cursor:
                total_benchmarks = (await cursor.fetchone())[0]

            async with db.execute("SELECT COUNT(*) FROM benchmark_results") as cursor:
                total_results = (await cursor.fetchone())[0]

            async with db.execute(
                "SELECT MAX(cached_at) FROM models WHERE source != 'manual'"
            ) as cursor:
                last_model_update = (await cursor.fetchone())[0]

            cache_size = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else None

            return CacheStatus(
                total_models=total_models,
                total_benchmarks=total_benchmarks,
                total_results=total_results,
                last_model_update=datetime.fromisoformat(last_model_update) if last_model_update else None,
                cache_size_mb=cache_size,
            )

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM benchmark_results")
            await db.execute("DELETE FROM models")
            await db.execute("DELETE FROM benchmarks")
            await db.commit()

    async def seed_sample_data(self) -> None:
        """Seed database with sample model data for offline testing."""
        sample_models = [
            ModelInfo(
                id="gpt-4o",
                model_id="openai/gpt-4o",
                name="GPT-4o",
                provider="openai",
                family="gpt-4",
                architecture=ArchitectureType.DENSE,
                context_length=128000,
                max_output_tokens=16384,
                supports_function_calling=True,
                supports_vision=True,
                supports_json_mode=True,
                input_price_per_million=2.5,
                output_price_per_million=10.0,
                source="sample",
            ),
            ModelInfo(
                id="gpt-4o-mini",
                model_id="openai/gpt-4o-mini",
                name="GPT-4o mini",
                provider="openai",
                family="gpt-4",
                architecture=ArchitectureType.DENSE,
                context_length=128000,
                max_output_tokens=16384,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_million=0.15,
                output_price_per_million=0.6,
                source="sample",
            ),
            ModelInfo(
                id="gpt-4-turbo",
                model_id="openai/gpt-4-turbo",
                name="GPT-4 Turbo",
                provider="openai",
                family="gpt-4",
                architecture=ArchitectureType.DENSE,
                context_length=128000,
                max_output_tokens=4096,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_million=10.0,
                output_price_per_million=30.0,
                source="sample",
            ),
            ModelInfo(
                id="claude-3-5-sonnet",
                model_id="anthropic/claude-3-5-sonnet",
                name="Claude 3.5 Sonnet",
                provider="anthropic",
                family="claude-3",
                architecture=ArchitectureType.DENSE,
                context_length=200000,
                max_output_tokens=8192,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_million=3.0,
                output_price_per_million=15.0,
                source="sample",
            ),
            ModelInfo(
                id="claude-3-opus",
                model_id="anthropic/claude-3-opus",
                name="Claude 3 Opus",
                provider="anthropic",
                family="claude-3",
                architecture=ArchitectureType.DENSE,
                context_length=200000,
                max_output_tokens=4096,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_million=15.0,
                output_price_per_million=75.0,
                source="sample",
            ),
            ModelInfo(
                id="claude-4-sonnet",
                model_id="anthropic/claude-4-sonnet",
                name="Claude 4 Sonnet",
                provider="anthropic",
                family="claude-4",
                architecture=ArchitectureType.DENSE,
                context_length=200000,
                max_output_tokens=64000,
                supports_function_calling=True,
                supports_vision=True,
                supports_reasoning=True,
                input_price_per_million=3.0,
                output_price_per_million=15.0,
                source="sample",
            ),
            ModelInfo(
                id="gemini-2.0-pro",
                model_id="google/gemini-2.0-pro",
                name="Gemini 2.0 Pro",
                provider="google",
                family="gemini",
                architecture=ArchitectureType.DENSE,
                context_length=2000000,
                max_output_tokens=8192,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_million=1.25,
                output_price_per_million=5.0,
                source="sample",
            ),
            ModelInfo(
                id="gemini-2.0-flash",
                model_id="google/gemini-2.0-flash",
                name="Gemini 2.0 Flash",
                provider="google",
                family="gemini",
                architecture=ArchitectureType.DENSE,
                context_length=1000000,
                max_output_tokens=8192,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_million=0.0,
                output_price_per_million=0.0,
                source="sample",
            ),
            ModelInfo(
                id="llama-3.1-405b",
                model_id="meta-llama/Llama-3.1-405B",
                name="Llama 3.1 405B",
                provider="meta",
                family="llama-3.1",
                architecture=ArchitectureType.DENSE,
                context_length=128000,
                max_output_tokens=4096,
                supports_function_calling=True,
                input_price_per_million=3.5,
                output_price_per_million=3.5,
                license="Llama 3.1",
                source="sample",
            ),
            ModelInfo(
                id="llama-3.1-70b",
                model_id="meta-llama/Llama-3.1-70B",
                name="Llama 3.1 70B",
                provider="meta",
                family="llama-3.1",
                architecture=ArchitectureType.DENSE,
                context_length=128000,
                max_output_tokens=4096,
                supports_function_calling=True,
                input_price_per_million=0.8,
                output_price_per_million=0.8,
                license="Llama 3.1",
                source="sample",
            ),
            ModelInfo(
                id="llama-4-scout",
                model_id="meta-llama/Llama-4-Scout",
                name="Llama 4 Scout",
                provider="meta",
                family="llama-4",
                architecture=ArchitectureType.MOE,
                total_params=109000000000,
                activated_params=17000000000,
                num_experts=16,
                context_length=10000000,
                max_output_tokens=32768,
                supports_vision=True,
                license="Llama 4",
                source="sample",
            ),
            ModelInfo(
                id="llama-4-maverick",
                model_id="meta-llama/Llama-4-Maverick",
                name="Llama 4 Maverick",
                provider="meta",
                family="llama-4",
                architecture=ArchitectureType.MOE,
                total_params=400000000000,
                activated_params=17000000000,
                num_experts=128,
                context_length=1000000,
                max_output_tokens=32768,
                supports_vision=True,
                license="Llama 4",
                source="sample",
            ),
            ModelInfo(
                id="mistral-large-3",
                model_id="mistralai/Mistral-Large-Instruct-24B",
                name="Mistral Large 3",
                provider="mistral",
                family="mistral",
                architecture=ArchitectureType.MOE,
                total_params=675000000000,
                activated_params=41000000000,
                context_length=128000,
                max_output_tokens=32768,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_million=2.0,
                output_price_per_million=8.0,
                source="sample",
            ),
            ModelInfo(
                id="mixtral-8x7b",
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                name="Mixtral 8x7B",
                provider="mistral",
                family="mixtral",
                architecture=ArchitectureType.MOE,
                total_params=46700000000,
                activated_params=12900000000,
                num_experts=8,
                context_length=32000,
                max_output_tokens=4096,
                supports_function_calling=True,
                input_price_per_million=0.24,
                output_price_per_million=0.24,
                license="Apache 2.0",
                source="sample",
            ),
            ModelInfo(
                id="qwen3-397b-a17b",
                model_id="Qwen/Qwen3-397B-A17B",
                name="Qwen3 397B-A17B",
                provider="alibaba",
                family="qwen3",
                architecture=ArchitectureType.MOE,
                total_params=397000000000,
                activated_params=17000000000,
                context_length=131072,
                max_output_tokens=4096,
                supports_function_calling=True,
                supports_vision=True,
                input_price_per_million=0.88,
                output_price_per_million=3.52,
                source="sample",
            ),
            ModelInfo(
                id="qwen2.5-72b",
                model_id="Qwen/Qwen2.5-72B",
                name="Qwen2.5 72B",
                provider="alibaba",
                family="qwen2.5",
                architecture=ArchitectureType.DENSE,
                context_length=32768,
                max_output_tokens=4096,
                supports_function_calling=True,
                input_price_per_million=0.9,
                output_price_per_million=0.9,
                source="sample",
            ),
        ]

        for model in sample_models:
            await self.save_model(model)

        sample_benchmarks = [
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
            BenchmarkInfo(
                id="chatbot-arena",
                name="Chatbot Arena",
                description="Human preference ranking of LLM chatbots through blind comparisons",
                category="human-preference",
                source_url="https://lmarena.ai",
                dataset_url="https://huggingface.co/datasets/lmsys/chatbot_arena_conversations",
            ),
        ]

        for benchmark in sample_benchmarks:
            await self.save_benchmark(benchmark)

    def _row_to_model(self, row: aiosqlite.Row) -> ModelInfo:
        """Convert database row to ModelInfo."""
        return ModelInfo(
            id=row["id"],
            model_id=row["model_id"],
            name=row["name"],
            provider=row["provider"],
            family=row["family"],
            total_params=row["total_params"],
            activated_params=row["activated_params"],
            architecture=row["architecture"],
            num_experts=row["num_experts"],
            attention_type=row["attention_type"],
            context_length=row["context_length"],
            max_output_tokens=row["max_output_tokens"],
            input_modalities=json.loads(row["input_modalities"]) if row["input_modalities"] else ["text"],
            output_modalities=json.loads(row["output_modalities"]) if row["output_modalities"] else ["text"],
            supports_function_calling=bool(row["supports_function_calling"]),
            supports_vision=bool(row["supports_vision"]),
            supports_reasoning=bool(row["supports_reasoning"]),
            supports_json_mode=bool(row["supports_json_mode"]),
            supports_streaming=bool(row["supports_streaming"]),
            input_price_per_million=row["input_price_per_million"],
            output_price_per_million=row["output_price_per_million"],
            release_date=datetime.fromisoformat(row["release_date"]) if row["release_date"] else None,
            knowledge_cutoff=datetime.fromisoformat(row["knowledge_cutoff"]) if row["knowledge_cutoff"] else None,
            training_tokens=row["training_tokens"],
            license=row["license"],
            huggingface_id=row["huggingface_id"],
            api_model_id=row["api_model_id"],
            cached_at=datetime.fromisoformat(row["cached_at"]),
            source=row["source"],
        )

    def _row_to_benchmark(self, row: aiosqlite.Row) -> BenchmarkInfo:
        """Convert database row to BenchmarkInfo."""
        return BenchmarkInfo(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            category=row["category"],
            source_url=row["source_url"],
            dataset_url=row["dataset_url"],
        )

    def _row_to_result(self, row: aiosqlite.Row) -> BenchmarkResult:
        """Convert database row to BenchmarkResult."""
        return BenchmarkResult(
            id=row["id"],
            model_id=row["model_id"],
            benchmark_id=row["benchmark_id"],
            score=row["score"],
            score_formatted=row["score_formatted"],
            rank=row["rank"],
            elo_rating=row["elo_rating"],
            evaluation_method=row["evaluation_method"],
            evaluated_at=datetime.fromisoformat(row["evaluated_at"]) if row["evaluated_at"] else None,
            source=row["source"],
        )