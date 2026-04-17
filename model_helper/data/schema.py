"""Database schema and initialization."""

import aiosqlite
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    family TEXT,
    total_params INTEGER,
    activated_params INTEGER,
    architecture TEXT,
    num_experts INTEGER,
    attention_type TEXT,
    context_length INTEGER,
    max_output_tokens INTEGER,
    input_modalities TEXT,
    output_modalities TEXT,
    supports_function_calling INTEGER DEFAULT 0,
    supports_vision INTEGER DEFAULT 0,
    supports_reasoning INTEGER DEFAULT 0,
    supports_json_mode INTEGER DEFAULT 0,
    supports_streaming INTEGER DEFAULT 1,
    input_price_per_million REAL,
    output_price_per_million REAL,
    release_date TEXT,
    knowledge_cutoff TEXT,
    training_tokens INTEGER,
    license TEXT,
    huggingface_id TEXT,
    api_model_id TEXT,
    cached_at TEXT NOT NULL,
    source TEXT DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT NOT NULL,
    source_url TEXT NOT NULL,
    dataset_url TEXT
);

CREATE TABLE IF NOT EXISTS benchmark_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    benchmark_id TEXT NOT NULL,
    score REAL,
    score_formatted TEXT,
    rank INTEGER,
    elo_rating REAL,
    evaluation_method TEXT,
    evaluated_at TEXT,
    source TEXT DEFAULT 'scraped',
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id),
    UNIQUE(model_id, benchmark_id)
);

CREATE INDEX IF NOT EXISTS idx_models_provider ON models(provider);
CREATE INDEX IF NOT EXISTS idx_models_family ON models(family);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
CREATE INDEX IF NOT EXISTS idx_results_model ON benchmark_results(model_id);
CREATE INDEX IF NOT EXISTS idx_results_benchmark ON benchmark_results(benchmark_id);
"""

DB_PATH = Path.home() / ".model_helper" / "cache.db"


async def init_db(db_path: Path | None = None) -> aiosqlite.Connection:
    """Initialize database with schema."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = await aiosqlite.connect(path)
    await conn.executescript(SCHEMA)
    await conn.commit()
    return conn


async def get_db(db_path: Path | None = None) -> aiosqlite.Connection:
    """Get database connection."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return await aiosqlite.connect(path)