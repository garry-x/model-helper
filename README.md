# Model Helper

LLM Model Information & Benchmark Lookup System

## Features

- **Provider Filtering**: Configure which model creators to track (e.g. OpenAI, Anthropic, DeepSeek). Only models from configured providers are fetched and displayed.
- **HuggingFace Hub Integration**: Fetches model metadata from HuggingFace Hub — the most comprehensive open model registry. Supports multiple mirror sites for reliable access.
- **Smart Mirror Fallback**: Automatic failover across 5+ mirror sites with connection preflight and intra-URL retry for maximum reliability.
- **Concurrent Fetching**: Parallel author requests via `asyncio` for fast cache updates even with many providers.
- **Benchmark Tracking**: Track results from MMLU, HellaSwag, ARC, TruthfulQA, Winogrande, GSM8K, HumanEval, and Chatbot Arena.
- **Fuzzy Search**: Find models using partial names (e.g., "gpt", "claude", "llama").
- **Local Caching**: SQLite-based cache for offline access.
- **CLI Interface**: Full command-line interface with live progress display.
- **Web Interface**: Beautiful web UI with Bootstrap dark theme and dynamic provider filtering.

## Installation

```bash
pip install -e .
```

## Configuration

Default configuration is stored in `model_helper/config.json` (shipped with the project):

| Section | Purpose |
|---------|---------|
| `providers` | Default model creators to track |
| `aliases` | Provider name aliases (creator → platform name) |
| `hf_authors` | HuggingFace author → provider name mapping |
| `hf_mirrors` | Mirror sites for HuggingFace API access |

User overrides are stored in `~/.model_helper/providers.json`.

## CLI Commands

### Provider management

```bash
model-helper provider-list                  # List configured providers
model-helper provider-add deepseek          # Add a provider
model-helper provider-remove bytedance      # Remove a provider
```

Once providers are configured, `cache-update` only fetches their models, and `list-models` only shows them. Use `--all` to see everything.

### Search for models

```bash
model-helper search gpt
model-helper search claude --limit 20
```

### View model details

```bash
model-helper model-info gpt-4
```

### List cached models

```bash
model-helper list-models                    # Defaults to configured providers
model-helper list-models --provider openai  # Filter by single provider
model-helper list-models --all              # Show all models regardless
```

### Benchmark commands

```bash
model-helper benchmark-list
model-helper benchmark-info mmlu
```

### Cache management

```bash
model-helper cache-status                   # Show cache statistics
model-helper cache-update                   # Fetch from HuggingFace + benchmarks
model-helper cache-update -v                # Verbose: per-page progress details
model-helper cache-update --timeout 120     # Custom timeout (default 60s)
model-helper cache-update -s benchmarks     # Update benchmarks only
model-helper cache-clear                    # Clear all cached data
model-helper cache-seed                     # Seed with sample data for offline use
```

### Start web interface

```bash
model-helper web
model-helper web --host 0.0.0.0 --port 8000
```

## Web Interface

Start the web server:

```bash
model-helper web
```

Then open http://localhost:8000 in your browser. The web UI respects your configured providers — use `?all=true` to see all cached models.

## Data Sources

- **HuggingFace Hub**: Primary source for model metadata (author, tags, parameters, architecture, license). Fetches text-generation models sorted by downloads.
- **HuggingFace Open LLM Leaderboard**: Benchmark definitions and results.
- **Chatbot Arena (LMSYS)**: Human preference rankings.

HuggingFace API access supports automatic mirror fallback. If the primary site is unreachable, the scraper tries mirrors in order (configured in `config.json` → `hf_mirrors`).

## Model Information Fields

- Model ID and name
- Provider (model creator — normalized from HuggingFace author)
- Model family (derived from architecture config)
- Total parameters (from safetensors metadata)
- Architecture type (Dense / MoE)
- Context length (from model config)
- Capabilities (Vision, Function Calling — from tags)
- License information (from tags or model card)

## Development

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## License

MIT
