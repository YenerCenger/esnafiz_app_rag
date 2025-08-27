# services/dependencies.py
from typing import Optional
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core.config import get_settings
from core.logging import logger

# External clients
import redis
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from google.cloud import firestore
from sentence_transformers import SentenceTransformer
import httpx
import hashlib

@dataclass
class ServiceManager:
    # Placeholders for connections/clients
    redis: Optional[object] = None
    qdrant: Optional[QdrantClient] = None
    firestore: Optional[firestore.Client] = None
    embed_model: Optional[SentenceTransformer] = None
    thread_pool: Optional[ThreadPoolExecutor] = None
    # simple in-service helpers
    @classmethod
    async def check_rate_limit(cls, key: str, max_per_min: int) -> bool:
        if not cls.redis:
            return True
        now = asyncio.get_event_loop().time()
        window_start = now - 60
        pipe = cls.redis.pipeline()
        pipe.zremrangebyscore(f"rl:{key}", "-inf", window_start)
        pipe.zadd(f"rl:{key}", {str(now): now})
        pipe.zcard(f"rl:{key}")
        pipe.expire(f"rl:{key}", 60)
        res = await asyncio.get_event_loop().run_in_executor(cls.thread_pool, pipe.execute)
        count = res[2] if isinstance(res, list) and len(res) >= 3 else 0
        return count <= max_per_min

    @classmethod
    async def cache_get(cls, key: str) -> Optional[str]:
        if not cls.redis:
            return None
        return await asyncio.get_event_loop().run_in_executor(cls.thread_pool, cls.redis.get, key)

    @classmethod
    async def cache_setex(cls, key: str, ttl: int, value: str) -> None:
        if not cls.redis:
            return None
        await asyncio.get_event_loop().run_in_executor(cls.thread_pool, lambda: cls.redis.setex(key, ttl, value))

    @classmethod
    def make_cache_key(cls, namespace: str, parts: list[str]) -> str:
        payload = ":".join(parts)
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    @classmethod
    async def call_llm(cls, prompt: str) -> str:
        settings = get_settings()
        payload = {
            "model": settings.default_llm,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": settings.llm_max_tokens,
                "temperature": settings.llm_temperature,
                "top_k": 40,
                "top_p": 0.9,
            },
        }
        timeout = httpx.Timeout(connect=10.0, read=float(settings.llm_timeout), write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
            res.raise_for_status()
            data = res.json()
            return (data.get("message") or {}).get("content", "").strip()

    # --- Firestore basit yardımcılar ---
    @classmethod
    async def get_store_items(cls, store_id: str, limit: int = 500) -> list[dict]:
        if not cls.firestore:
            return []
        loop = asyncio.get_event_loop()
        query = cls.firestore.collection("items").where("storeId", "==", store_id).limit(limit)
        docs = await loop.run_in_executor(cls.thread_pool, lambda: list(query.stream()))
        items = []
        for d in docs:
            try:
                obj = d.to_dict() or {}
                obj["id"] = d.id
                # güvenli tip dönüşümleri
                obj["stock"] = int(obj.get("stock", 0) or 0)
                obj["price"] = float(obj.get("price", 0) or 0)
                obj["name"] = str(obj.get("name", "") or "").strip()
                items.append(obj)
            except Exception:
                continue
        return items

    @classmethod
    async def get_store_sales(cls, store_id: str, days: int = 30, limit: int = 3000) -> list[dict]:
        if not cls.firestore:
            return []
        from datetime import datetime, timedelta
        loop = asyncio.get_event_loop()
        start_date = datetime.utcnow() - timedelta(days=days)
        query = (
            cls.firestore.collection("sold_products")
            .where("storeId", "==", store_id)
            .where("created_at", ">=", start_date)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        docs = await loop.run_in_executor(cls.thread_pool, lambda: list(query.stream()))
        sales = []
        for d in docs:
            try:
                obj = d.to_dict() or {}
                obj["id"] = d.id
                obj["quantity"] = int(obj.get("quantity", 1) or 1)
                obj["price"] = float(obj.get("price", 0) or 0)
                obj["product_name"] = str(obj.get("product_name", "Unknown") or "Unknown").strip()
                sales.append(obj)
            except Exception:
                continue
        return sales

    @classmethod
    async def initialize_all(cls):
        settings = get_settings()
        logger.info("Initializing services")

        # Thread pool
        if cls.thread_pool is None:
            cls.thread_pool = ThreadPoolExecutor(max_workers=settings.max_workers)

        loop = asyncio.get_event_loop()

        async def init_embed():
            logger.info("Loading embedding model", model=settings.embed_model)
            cls.embed_model = await loop.run_in_executor(
                cls.thread_pool, lambda: SentenceTransformer(settings.embed_model)
            )

        async def init_qdrant():
            logger.info("Connecting Qdrant", url=settings.qdrant_url)
            cls.qdrant = QdrantClient(url=settings.qdrant_url, timeout=30)
            # Ensure collection exists
            collection_name = settings.collection_name
            try:
                await loop.run_in_executor(cls.thread_pool, lambda: cls.qdrant.get_collection(collection_name))
            except Exception:
                await loop.run_in_executor(
                    cls.thread_pool,
                    lambda: cls.qdrant.recreate_collection(
                        collection_name,
                        vectors_config=VectorParams(
                            size=(cls.embed_model.get_sentence_embedding_dimension() if cls.embed_model else 384),
                            distance=Distance.COSINE,
                        ),
                    ),
                )

        async def init_redis():
            logger.info("Connecting Redis", url=settings.redis_url)
            cls.redis = redis.from_url(**settings.get_redis_config())
            await loop.run_in_executor(cls.thread_pool, cls.redis.ping)

        async def init_firestore():
            logger.info("Connecting Firestore", project=settings.project_id)
            cls.firestore = firestore.Client(project=settings.project_id)
            # test
            await loop.run_in_executor(
                cls.thread_pool, lambda: list(cls.firestore.collection("stores").limit(1).stream())
            )

        await asyncio.gather(init_embed(), init_qdrant(), init_redis(), init_firestore())
        logger.info("Services initialized")

    @classmethod
    async def cleanup_all(cls):
        logger.info("Cleaning up services")
        tasks = []
        loop = asyncio.get_event_loop()
        if cls.redis:
            tasks.append(loop.run_in_executor(cls.thread_pool, cls.redis.close))
        if cls.qdrant:
            tasks.append(loop.run_in_executor(cls.thread_pool, cls.qdrant.close))
        if cls.thread_pool:
            tasks.append(loop.run_in_executor(None, cls.thread_pool.shutdown, True))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Services cleaned up")

# Dependency injection
_singleton: Optional[ServiceManager] = ServiceManager()

def get_service_manager() -> ServiceManager:
    # In real app, consider per-request context if needed
    return _singleton
