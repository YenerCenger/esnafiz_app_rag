# api/endpoints/health.py
from fastapi import APIRouter
from datetime import datetime
import asyncio

from services.dependencies import get_service_manager
from core.config import get_settings

router = APIRouter(tags=["health"])

@router.get("/health", summary="Liveness/Readiness")
async def health():
    sm = get_service_manager()
    settings = get_settings()
    services = {
        "redis": "unknown",
        "qdrant": "unknown",
        "firestore": "unknown",
        "embedding": "unknown",
    }

    # Quick checks (non-failing)
    try:
        if sm.redis:
            await asyncio.get_event_loop().run_in_executor(sm.thread_pool, sm.redis.ping)
            services["redis"] = "healthy"
        else:
            services["redis"] = "uninitialized"
    except Exception:
        services["redis"] = "unhealthy"

    try:
        if sm.qdrant:
            await asyncio.get_event_loop().run_in_executor(
                sm.thread_pool, lambda: sm.qdrant.get_collection(settings.collection_name)
            )
            services["qdrant"] = "healthy"
        else:
            services["qdrant"] = "uninitialized"
    except Exception:
        services["qdrant"] = "unhealthy"

    try:
        if sm.firestore:
            # simple list to test
            await asyncio.get_event_loop().run_in_executor(
                sm.thread_pool, lambda: list(sm.firestore.collections())
            )
            services["firestore"] = "healthy"
        else:
            services["firestore"] = "uninitialized"
    except Exception:
        services["firestore"] = "unhealthy"

    services["embedding"] = "healthy" if sm.embed_model else "uninitialized"

    overall = "healthy" if all(v in ("healthy", "uninitialized") for v in services.values()) else "degraded"

    return {
        "service": "Esnafiz AI Assistant API",
        "status": overall,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": services,
    }
