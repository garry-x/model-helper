"""CLI entry point using Typer."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from model_helper import config
from model_helper.cache.manager import CacheManager
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
        active_providers = config.get_providers() or None
        if active_providers:
            active_providers = config.resolve_providers(active_providers)
        models = await cache.list_models(providers=active_providers, limit=500)
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

        def _price(p: float | None) -> str:
            return f"${p}/M" if p is not None else "N/A"

        def _ctx(c: int | None) -> str:
            return f"{c:,}" if c is not None else "N/A"

        def _yn(b: bool) -> str:
            return "✓" if b else "✗"

        panel = Panel(
            f"[bold]Provider:[/bold] {model.provider}\n"
            f"[bold]Model ID:[/bold] {model.model_id or 'N/A'}\n"
            f"[bold]Family:[/bold] {model.family or 'N/A'}\n"
            f"[bold]Total Params:[/bold] {model.total_params or 'N/A'}\n"
            f"[bold]Activated Params:[/bold] {model.activated_params or 'N/A'}\n"
            f"[bold]Architecture:[/bold] {model.architecture.value if model.architecture else 'N/A'}\n"
            f"[bold]Context Length:[/bold] {_ctx(model.context_length)}\n"
            f"[bold]Max Output:[/bold] {model.max_output_tokens or 'N/A'}\n"
            f"[bold]Input Price:[/bold] {_price(model.input_price_per_million)}\n"
            f"[bold]Output Price:[/bold] {_price(model.output_price_per_million)}\n"
            f"[bold]Function Calling:[/bold] {_yn(model.supports_function_calling)}\n"
            f"[bold]Vision:[/bold] {_yn(model.supports_vision)}\n"
            f"[bold]JSON Mode:[/bold] {_yn(model.supports_json_mode)}\n"
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
    all: bool = typer.Option(False, "--all", "-a", help="Show all models regardless of configured providers"),
):
    """List cached models (defaults to configured providers if set)."""
    async def _list():
        cache = get_cache()
        await cache.init()

        active_providers = None if all else (config.get_providers() or None)
        if provider:
            active_providers = [provider]
        elif active_providers:
            active_providers = config.resolve_providers(active_providers)

        models = await cache.list_models(providers=active_providers, limit=limit)

        if not models:
            console.print("[yellow]No models in cache. Run 'cache update' first.[/yellow]")
            return

        title = f"Cached Models ({len(models)})"
        if active_providers:
            title += f" [providers: {', '.join(active_providers)}]"

        table = Table(title=title)
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
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="green", no_wrap=True)
        table.add_column("Category", style="yellow")
        table.add_column("Source URL", style="blue", max_width=60)

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
            f"[bold]Cache Size:[/bold] {f'{status.cache_size_mb:.2f} MB' if status.cache_size_mb else 'N/A'}",
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
    """Update cache with latest data from online sources.

    If providers are configured via 'provider-add', only models from those
    providers will be saved. Otherwise all models are saved.
    """
    async def _update():
        cache = get_cache()
        await cache.init()
        providers = config.get_providers()
        active_providers = config.resolve_providers(providers) if providers else []

        if providers:
            console.print(f"[dim]Configured providers: {', '.join(providers)}[/dim]")

        sources = [source] if source else ["litellm", "modeldb", "benchmarks"]

        for src in sources:
            console.print(f"[cyan]Updating from {src}...[/cyan]")

            try:
                if src == "litellm":
                    async with LiteLLMScraper() as scraper:
                        models = await scraper.fetch_models()
                        if active_providers:
                            models = [m for m in models if m.provider.lower() in active_providers]
                        for model in models:
                            await cache.save_model(model)
                        msg = f"✓ Added {len(models)} models from LiteLLM"
                        if active_providers:
                            msg += " (filtered by configured providers)"
                        console.print(f"[green]{msg}[/green]")

                elif src == "modeldb":
                    async with ModelDBScraper() as scraper:
                        models = await scraper.fetch_models()
                        if active_providers:
                            models = [m for m in models if m.provider.lower() in active_providers]
                        for model in models:
                            await cache.save_model(model)
                        msg = f"✓ Added {len(models)} models from ModelDB"
                        if active_providers:
                            msg += " (filtered by configured providers)"
                        console.print(f"[green]{msg}[/green]")

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
def cache_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Clear all cached data."""
    if not yes:
        confirm = typer.confirm("Are you sure you want to clear all cached data?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit()

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
def provider_add(
    name: str = typer.Argument(..., help="Provider name to add (e.g. openai, anthropic)"),
):
    """Add a provider to the configured list.

    Once providers are configured, cache-update will only fetch models
    from these providers, and list-models will only show them.
    """
    if config.add_provider(name):
        console.print(f"[green]✓ Added provider '{name.lower()}'[/green]")
    else:
        console.print(f"[yellow]Provider '{name.lower()}' is already configured[/yellow]")


@app.command()
def provider_remove(
    name: str = typer.Argument(..., help="Provider name to remove"),
):
    """Remove a provider from the configured list."""
    if config.remove_provider(name):
        console.print(f"[green]✓ Removed provider '{name.lower()}'[/green]")
    else:
        console.print(f"[yellow]Provider '{name.lower()}' is not in the configured list[/yellow]")


@app.command()
def provider_list():
    """List all configured providers."""
    providers = config.get_providers()
    is_default = not config._config_path().exists()

    if not providers:
        console.print("[dim]No providers configured. All providers will be used.[/dim]")
        console.print("Use 'model-helper provider-add <name>' to add providers.")
        return

    title = f"Configured Providers ({len(providers)})"
    if is_default:
        title += " [dim]— using defaults[/dim]"
    table = Table(title=title)
    table.add_column("Provider", style="green")
    for p in providers:
        table.add_row(p)

    if is_default:
        console.print("[dim]Defaults are active because no config file exists.[/dim]")
        console.print("[dim]Add or remove providers to create a custom config.[/dim]")
    console.print(table)


@app.command()
def web(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the web interface."""
    import uvicorn

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