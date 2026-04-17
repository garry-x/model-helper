"""Fuzzy search engine using rapidfuzz."""

from rapidfuzz import fuzz, process

from model_helper.data.models import ModelInfo, SearchResult


class SearchEngine:
    """Fuzzy search for model names."""

    def __init__(self, threshold: int = 60):
        self.threshold = threshold

    def search(
        self,
        query: str,
        models: list[ModelInfo],
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search models using fuzzy matching."""
        if not query or not models:
            return []

        model_names = [(m.id, m) for m in models]
        choices = [m.id for m in models]

        matches = process.extract(
            query,
            choices,
            scorer=fuzz.WRatio,
            limit=limit,
        )

        results = []
        for match_id, score, _ in matches:
            if score >= self.threshold:
                model = next(m for m_id, m in model_names if m_id == match_id)
                matched_on = "id" if model.id.lower() == query.lower() else "name"
                if model.name.lower() == query.lower():
                    matched_on = "name"
                if model.model_id and model.model_id.lower() == query.lower():
                    matched_on = "model_id"
                results.append(SearchResult(model=model, score=score, matched_on=matched_on))

        return results

    def search_by_name(self, query: str, models: list[ModelInfo], limit: int = 10) -> list[SearchResult]:
        """Search specifically by model name."""
        return self.search(query, models, limit)

    def search_by_provider(self, provider: str, models: list[ModelInfo]) -> list[ModelInfo]:
        """Filter models by provider."""
        return [m for m in models if m.provider.lower() == provider.lower()]

    def search_by_architecture(self, arch: str, models: list[ModelInfo]) -> list[ModelInfo]:
        """Filter models by architecture type."""
        return [
            m
            for m in models
            if m.architecture and m.architecture.value.lower() == arch.lower()
        ]

    def search_moe_models(self, models: list[ModelInfo]) -> list[ModelInfo]:
        """Get all MoE models."""
        return [m for m in models if m.architecture and m.architecture.value == "moe"]

    def search_vision_models(self, models: list[ModelInfo]) -> list[ModelInfo]:
        """Get all vision-capable models."""
        return [m for m in models if m.supports_vision]