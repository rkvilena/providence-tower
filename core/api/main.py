from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.api.routes.health import router as health_router
from core.api.routes.rag import router as rag_router
from core.api.service import RagService

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan handler – initialises / tears down shared resources."""
    # --- startup ---
    svc = RagService()
    svc.startup()
    app.state.rag_service = svc
    LOGGER.info("RagService initialised.")

    yield

    # --- shutdown ---
    svc.shutdown()
    LOGGER.info("RagService shut down.")


def create_app(
    *,
    title: str = "Providence Tower v2 API",
    version: str = "2.0.0",
    docs_url: str = "/docs",
    redoc_url: str = "/redoc",
    openapi_url: str = "/openapi.json",
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Build and return a configured FastAPI application instance."""
    app = FastAPI(
        title=title,
        version=version,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    # CORS – allow all origins in development; tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(rag_router)

    return app
