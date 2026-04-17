# Model Helper

LLM Model Information & Benchmark Lookup System

## Features

- **Model Information Database**: Comprehensive data on LLM models including parameters, architecture (MoE/Dense), context length, pricing, and capabilities
- **Benchmark Tracking**: Track results from MMLU, HellaSwag, ARC, TruthfulQA, Winogrande, GSM8K, HumanEval, and Chatbot Arena
- **Fuzzy Search**: Find models using partial names (e.g., "gpt", "claude", "llama")
- **Multiple Data Sources**: Fetches from LiteLLM, ModelDB, and HuggingFace
- **Local Caching**: SQLite-based cache for offline access
- **CLI Interface**: Full command-line interface for all operations
- **Web Interface**: Beautiful web UI with Bootstrap dark theme

## Installation

```bash
pip install -e .
```

## CLI Commands

### Search for models
```bash
model-helper search gpt
model-helper search claude --limit 20
```

### View model details
```bash
model-helper model-info gpt-4
```

### List all cached models
```bash
model-helper list-models
model-helper list-models --provider openai
```

### Benchmark commands
```bash
model-helper benchmark-list
model-helper benchmark-info mmlu
```

### Cache management
```bash
model-helper cache-status
model-helper cache-update
model-helper cache-update --source litellm
model-helper cache-clear
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

Then open http://localhost:8000 in your browser.

## Data Sources

- **LiteLLM**: Comprehensive model pricing and context data
- **ModelDB**: Free API for model metadata
- **HuggingFace Open LLM Leaderboard**: Academic benchmark results
- **Chatbot Arena**: Human preference rankings

## Model Information Fields

- Total parameters
- Activated parameters (for MoE models)
- Architecture type (Dense/MoE)
- Number of experts (for MoE)
- Context length
- Max output tokens
- Input/Output pricing
- Capabilities (Vision, Function Calling, JSON Mode)
- License information

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