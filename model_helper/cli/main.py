"""CLI entry point using Typer."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from model_helper.cache.manager import CacheManager
from model_helper.data.models import BenchmarkInfo
from model_helper.data.models import BenchmarkInfo, ModelInfo
from model_helper.scrapers.base import (
    ChatbotArenaScraper,
    LiteLLMScraper,
    ModelDBScraper,
    OpenLLMLeaderboardScraper,
)
from model_helper.search.engine import SearchEngine

app = typer.Typer(help="LLM Model Information & Benchmark Lookup System")
console = Console()

cache_manager: Optional[CacheManager] = None
search_engine: Optional[SearchEngine] = None


def get_cache() -> CacheManager:
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager()
    return cache_manager


def get_search() -> SearchEngine:
    global search_engine
    if search_engine is None:
        search_engine = SearchEngine()
    return search_engine


@app.command()
def search(
    query: str = typer.Argument(..., help="Model name to search for"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    threshold: int = typer.Option(60, "--threshold", "-t", help="Minimum match score"),
):
    """Search for models by name (fuzzy search)."""
    async def _search():
        cache = get_cache()
        await cache.init()
        models = await cache.list_models(limit=500)
        search_eng = get_search()
        search_eng.threshold = threshold
        results = search_eng.search(query, models, limit)

        if not results:
            console.print(f"[yellow]No models found matching '{query}'[/yellow]")
            return

        table = Table(title=f"Search Results for '{query}'")
        table.add_column("Name", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Context", style="magenta")
        table.add_column("Score", style="yellow")

        for result in results:
            model = result.model
            ctx = f"{model.context_length:,}" if model.context_length else "N/A"
            table.add_row(model.name, model.provider, ctx, f"{result.score:.1f}")

        console.print(table)

    asyncio.run(_search())


@app.command()
def model_info(
    name: str = typer.Argument(..., help="Model name or ID"),
):
    """Show detailed information about a model."""
    async def _model_info():
        cache = get_cache()
        await cache.init()
        model = await cache.get_model(name)

        if not model:
            console.print(f"[red]Model '{name}' not found[/red]")
            return

        results = await cache.get_results_for_model(model.id)

        panel = Panel(
            f"[bold]Provider:[/bold] {model.provider}\n"
            f"[bold]Family:[/bold] {model.family or 'N/A'}\n"
            f"[bold]Total Params:[/bold] {model.total_params or 'N/A'}\n"
            f"[bold]Activated Params:[/bold] {model.activated_params or 'N/A'}\n"
            f"[bold]Architecture:[/bold] {model.architecture.value if model.architecture else 'N/A'}\n"
            f"[bold]Context Length:[/bold] {model.context_length or 'N/A'}\n"
            f"[bold]Max Output:[/bold] {model.max_output_tokens or 'N/A'}\n"
            f"[bold]Input Price:[/bold] ${model.input_price_per_million or 'N/A'}/M\n"
            f"[bold]Output Price:[/bold] ${model.output_price_per_million or 'N/A'}/M\n"
            f"[bold]Function Calling:[/bold] {'✓' if model.supports_function_calling else '✗'}\n"
            f"[bold]Vision:[/bold] {'✓' if model.supports_vision else '✗'}\n"
            f"[bold]JSON Mode:[/bold] {'✓' if model.supports_json_mode else '✗'}\n"
            f"[bold]License:[/bold] {model.license or 'N/A'}\n"
            f"[bold]Source:[/bold] {model.source}",
            title=f"Model: {model.name}",
            border_style="cyan",
        )
        console.print(panel)

        if results:
            table = Table(title="Benchmark Results")
            table.add_column("Benchmark", style="cyan")
            table.add_column("Score", style="green")
            table.add_column("Rank", style="yellow")
            for r in results:
                table.add_row(
                    r.benchmark_id,
                    r.score_formatted or str(r.score) if r.score else "N/A",
                    str(r.rank) if r.rank else "N/A",
                )
            console.print(table)

    asyncio.run(_model_info())


@app.command()
def list_models(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Filter by provider"),
    limit: int = typer.Option(50, "--limit", "-n", help="Number of models to show"),
):
    """List all cached models."""
    async def _list():
        cache = get_cache()
        await cache.init()
        models = await cache.list_models(provider=provider, limit=limit)

        if not models:
            console.print("[yellow]No models in cache. Run 'cache update' first.[/yellow]")
            return

        table = Table(title=f"Cached Models ({len(models)})")
        table.add_column("Name", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Context", style="magenta")
        table.add_column("Source", style="dim")

        for model in models:
            ctx = f"{model.context_length:,}" if model.context_length else "N/A"
            table.add_row(model.name, model.provider, ctx, model.source)

        console.print(table)

    asyncio.run(_list())


@app.command()
def benchmark_list():
    """List all available benchmarks."""
    async def _list():
        cache = get_cache()
        await cache.init()
        benchmarks = await cache.list_benchmarks()

        if not benchmarks:
            console.print("[yellow]No benchmarks in cache. Run 'cache update' first.[/yellow]")
            return

        table = Table(title=f"Available Benchmarks ({len(benchmarks)})")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Category", style="yellow")
        table.add_column("Source URL", style="blue", no_wrap=True)

        for b in benchmarks:
            table.add_row(b.id, b.name, b.category, b.source_url)

        console.print(table)

    asyncio.run(_list())


@app.command()
def benchmark_info(
    name: str = typer.Argument(..., help="Benchmark ID"),
):
    """Show detailed information about a benchmark."""
    async def _info():
        cache = get_cache()
        await cache.init()
        benchmark = await cache.get_benchmark(name)

        if not benchmark:
            console.print(f"[red]Benchmark '{name}' not found[/red]")
            return

        results = await cache.get_results_for_benchmark(name, limit=20)

        panel = Panel(
            f"[bold]Description:[/bold] {benchmark.description}\n"
            f"[bold]Category:[/bold] {benchmark.category}\n"
            f"[bold]Source URL:[/bold] {benchmark.source_url}\n"
            f"[bold]Dataset URL:[/bold] {benchmark.dataset_url or 'N/A'}",
            title=f"Benchmark: {benchmark.name}",
            border_style="cyan",
        )
        console.print(panel)

        if results:
            table = Table(title=f"Top Results for {benchmark.name}")
            table.add_column("Model", style="cyan")
            table.add_column("Score", style="green")
            table.add_column("Rank", style="yellow")
            for r in results:
                model = await cache.get_model(r.model_id)
                model_name = model.name if model else r.model_id
                table.add_row(
                    model_name,
                    r.score_formatted or str(r.score) if r.score else "N/A",
                    str(r.rank) if r.rank else "N/A",
                )
            console.print(table)

    asyncio.run(_info())


@app.command()
def cache_status():
    """Show cache status and statistics."""
    async def _status():
        cache = get_cache()
        await cache.init()
        status = await cache.get_status()

        panel = Panel(
            f"[bold]Total Models:[/bold] {status.total_models}\n"
            f"[bold]Total Benchmarks:[/bold] {status.total_benchmarks}\n"
            f"[bold]Total Results:[/bold] {status.total_results}\n"
            f"[bold]Last Model Update:[/bold] {status.last_model_update or 'Never'}\n"
            f"[bold]Cache Size:[/bold] {status.cache_size_mb:.2f} MB" if status.cache_size_mb else "N/A",
            title="Cache Status",
            border_style="green",
        )
        console.print(panel)

    asyncio.run(_status())


@app.command()
def cache_update(
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Update specific source (litellm, modeldb, benchmarks)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
):
    """Update cache with latest data from online sources."""
    async def _update():
        cache = get_cache()
        await cache.init()

        sources = [source] if source else ["litellm", "modeldb", "benchmarks"]

        for src in sources:
            console.print(f"[cyan]Updating from {src}...[/cyan]")

            try:
                if src == "litellm":
                    async with LiteLLMScraper() as scraper:
                        models = await scraper.fetch_models()
                        for model in models:
                            await cache.save_model(model)
                        console.print(f"[green]✓ Added {len(models)} models from LiteLLM[/green]")

                elif src == "modeldb":
                    async with ModelDBScraper() as scraper:
                        models = await scraper.fetch_models()
                        for model in models:
                            await cache.save_model(model)
                        console.print(f"[green]✓ Added {len(models)} models from ModelDB[/green]")

                elif src == "benchmarks":
                    all_benchmarks = []
                    async with OpenLLMLeaderboardScraper() as scraper:
                        benchmarks = await scraper.fetch_benchmarks()
                        all_benchmarks.extend(benchmarks)
                    async with ChatbotArenaScraper() as scraper:
                        benchmarks = await scraper.fetch_benchmarks()
                        all_benchmarks.extend(benchmarks)

                    for benchmark in all_benchmarks:
                        await cache.save_benchmark(benchmark)
                    console.print(f"[green]✓ Added {len(all_benchmarks)} benchmarks[/green]")

            except Exception as e:
                console.print(f"[red]✗ Failed to update from {src}: {e}[/red]")

        console.print("[bold green]Cache update complete![/bold green]")

    asyncio.run(_update())


@app.command()
def cache_clear():
    """Clear all cached data."""
    confirm = typer.confirm("Are you sure you want to clear all cached data?")
    if not confirm:
        console.print("[yellow]Cancelled[/yellow]")
        return

    async def _clear():
        cache = get_cache()
        await cache.init()
        await cache.clear_cache()
        console.print("[green]Cache cleared successfully[/green]")

    asyncio.run(_clear())


@app.command()
def cache_seed():
    """Seed cache with sample data for offline testing."""
    async def _seed():
        cache = get_cache()
        await cache.init()
        await cache.seed_sample_data()
        console.print("[green]Sample data seeded successfully![/green]")

    asyncio.run(_seed())


@app.command()
def web(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the web interface."""
    import uvicorn

    from model_helper.web.app import create_app

    app = create_app()
    console.print(f"[green]Starting web server at http://{host}:{port}[/green]")
    uvicorn.run(
        "model_helper.web.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


if __name__ == "__main__":
    app()