# services/retrieval/hybrid.py
# Basit hibrit arama: Şimdilik yalnızca vektör araması (Qdrant) uygular.
# Gerekirse ileride BM25 ve RRF eklenebilir.

from typing import List, Dict, Any, Optional
import asyncio

from qdrant_client.models import Filter, FieldCondition, MatchValue

from core.config import get_settings


def _embed_text(sm, text: str) -> List[float]:
    if not sm.embed_model:
        raise RuntimeError("Embedding modeli hazır değil")
    embeddings = sm.embed_model.encode([f"query: {text}"], normalize_embeddings=True, show_progress_bar=False)
    return embeddings[0].tolist()


async def _vector_search(sm, query_vector: List[float], k: int, store_id: Optional[str]) -> List[Dict[str, Any]]:
    if not sm.qdrant:
        return []
    settings = get_settings()

    qfilter = None
    if store_id:
        qfilter = Filter(must=[FieldCondition(key="store_id", match=MatchValue(value=store_id))])

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        sm.thread_pool,
        lambda: sm.qdrant.search(
            collection_name=settings.collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=qfilter,
        ),
    )

    parsed: List[Dict[str, Any]] = []
    for r in results or []:
        payload = getattr(r, "payload", {}) or {}
        text = payload.get("text") or payload.get("chunk") or ""
        parsed.append({
            "id": getattr(r, "id", None),
            "score": getattr(r, "score", 0.0),
            "text": text[:1000],
            "payload": {
                "doc_id": payload.get("doc_id"),
                "collection": payload.get("collection"),
                "store_id": payload.get("store_id"),
            },
        })
    return parsed


async def hybrid_search(sm, query: str, k: int = 8, store_id: Optional[str] = None) -> List[Dict[str, Any]]:
    # Embed et ve Qdrant vektör araması yap
    loop = asyncio.get_event_loop()
    query_vec = await loop.run_in_executor(sm.thread_pool, _embed_text, sm, query)
    vector_results = await _vector_search(sm, query_vec, k, store_id)

    # İleride: BM25 + RRF ile zenginleştirilebilir
    return vector_results
