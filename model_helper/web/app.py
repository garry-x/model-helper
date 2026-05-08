"""FastAPI web application."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from model_helper import config
from model_helper.cache.manager import CacheManager
from model_helper.search.engine import SearchEngine

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
cache_manager = CacheManager()
search_engine = SearchEngine()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Model Helper",
        description="LLM Model Information & Benchmark Lookup System",
        version="0.1.0",
    )

    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.on_event("startup")
    async def startup():
        await cache_manager.init()

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        status = await cache_manager.get_status()
        p = config.get_providers()
        active_providers = config.resolve_providers(p) if p else None
        models = await cache_manager.list_models(providers=active_providers, limit=6)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "models": models, "status": status},
        )

    @app.get("/models", response_class=HTMLResponse)
    async def models(
        request: Request,
        provider: Optional[str] = Query(None),
        page: int = Query(1, ge=1),
        show_all: bool = Query(False, alias="all"),
    ):
        limit = 20
        offset = (page - 1) * limit
        active_providers = None
        if provider:
            active_providers = [provider]
        elif not show_all:
            p = config.get_providers()
            active_providers = config.resolve_providers(p) if p else None
        models = await cache_manager.list_models(providers=active_providers, limit=limit, offset=offset)
        status = await cache_manager.get_status()
        configured_providers = config.get_providers()
        return templates.TemplateResponse(
            "models/list.html",
            {
                "request": request, "models": models, "page": page,
                "provider": provider, "status": status,
                "configured_providers": configured_providers,
            },
        )

    @app.get("/models/{model_id}", response_class=HTMLResponse)
    async def model_detail(request: Request, model_id: str):
        model = await cache_manager.get_model(model_id)
        if not model:
            return templates.TemplateResponse(
                "404.html",
                {"request": request, "message": f"Model '{model_id}' not found"},
                status_code=404,
            )
        results = await cache_manager.get_results_for_model(model_id)
        status = await cache_manager.get_status()
        return templates.TemplateResponse(
            "models/detail.html",
            {"request": request, "model": model, "results": results, "status": status},
        )

    @app.get("/benchmarks", response_class=HTMLResponse)
    async def benchmarks(request: Request):
        benchmarks = await cache_manager.list_benchmarks()
        status = await cache_manager.get_status()
        return templates.TemplateResponse(
            "benchmarks/index.html",
            {"request": request, "benchmarks": benchmarks, "status": status},
        )

    @app.get("/search", response_class=HTMLResponse)
    async def search_page(request: Request, q: str = Query(...)):
        p = config.get_providers()
        active_providers = config.resolve_providers(p) if p else None
        models = await cache_manager.list_models(providers=active_providers, limit=500)
        results = search_engine.search(q, models, limit=20)
        status = await cache_manager.get_status()
        return templates.TemplateResponse(
            "search.html",
            {"request": request, "results": results, "query": q, "status": status},
        )

    @app.get("/api/search")
    async def api_search(q: str = Query(...), limit: int = Query(10, ge=1, le=50)):
        p = config.get_providers()
        active_providers = config.resolve_providers(p) if p else None
        models = await cache_manager.list_models(providers=active_providers, limit=500)
        results = search_engine.search(q, models, limit=limit)
        return {
            "query": q,
            "count": len(results),
            "results": [
                {
                    "id": r.model.id,
                    "name": r.model.name,
                    "provider": r.model.provider,
                    "context_length": r.model.context_length,
                    "score": r.score,
                }
                for r in results
            ],
        }

    @app.get("/api/models")
    async def api_models(
        provider: Optional[str] = None,
        limit: int = Query(20, ge=1, le=100),
        show_all: bool = Query(False, alias="all"),
    ):
        active_providers = None
        if provider:
            active_providers = [provider]
        elif not show_all:
            p = config.get_providers()
            active_providers = config.resolve_providers(p) if p else None
        models = await cache_manager.list_models(providers=active_providers, limit=limit)
        return {
            "count": len(models),
            "models": [m.model_dump() for m in models],
        }

    @app.get("/api/benchmarks")
    async def api_benchmarks():
        benchmarks = await cache_manager.list_benchmarks()
        return {
            "count": len(benchmarks),
            "benchmarks": [b.model_dump() for b in benchmarks],
        }

    @app.get("/api/status")
    async def api_status():
        status = await cache_manager.get_status()
        return status.model_dump()

    return app